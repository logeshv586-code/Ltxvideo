"""Automatic long-form scene planning and low-VRAM sequential generation.

The customer supplies one paragraph. This module turns it into chronological
short LTX shots, generates them one at a time, carries the previous ending
frame into the next shot, and concatenates everything without holding a whole
multi-minute video in RAM/VRAM.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

from PIL import Image

from config import DEFAULT_FPS, OUTPUTS_DIR
from engine.video_processor import concatenate_videos_streaming, extract_last_frame

Progress = Callable[[str, float], None] | None
MAX_LONGFORM_SECONDS = 300
MAX_LONGFORM_SCENES = 96


@dataclass(frozen=True)
class RenderProfile:
    key: str
    label: str
    landscape: tuple[int, int]
    portrait: tuple[int, int]
    square: tuple[int, int]
    frames_per_scene: int
    inference_steps: int
    guidance_scale: float
    delivery_long_edge: int

    @property
    def scene_seconds(self) -> float:
        return self.frames_per_scene / DEFAULT_FPS

    def size_for(self, aspect: str) -> tuple[int, int]:
        if aspect == "9:16":
            return self.portrait
        if aspect == "1:1":
            return self.square
        return self.landscape


QUALITY_PROFILES: dict[str, RenderProfile] = {
    "Fast": RenderProfile(
        key="fast",
        label="Fast • best for drafts and long stories",
        landscape=(384, 224),
        portrait=(224, 384),
        square=(320, 320),
        frames_per_scene=97,
        inference_steps=12,
        guidance_scale=3.0,
        delivery_long_edge=1280,
    ),
    "Balanced": RenderProfile(
        key="balanced",
        label="Balanced • recommended for RTX 4050 6 GB",
        landscape=(512, 288),
        portrait=(288, 512),
        square=(384, 384),
        frames_per_scene=121,
        inference_steps=16,
        guidance_scale=3.0,
        delivery_long_edge=1920,
    ),
    "Quality": RenderProfile(
        key="quality",
        label="Quality • slower, more native detail",
        landscape=(576, 320),
        portrait=(320, 576),
        square=(448, 448),
        frames_per_scene=121,
        inference_steps=20,
        guidance_scale=3.2,
        delivery_long_edge=1920,
    ),
}

DURATION_SECONDS = {
    "Auto from story": None,
    "15 seconds": 15,
    "30 seconds": 30,
    "1 minute": 60,
    "2 minutes": 120,
    "3 minutes": 180,
    "4 minutes": 240,
    "5 minutes": 300,
}

ASPECT_LABELS = {
    "YouTube / Landscape (16:9)": "16:9",
    "Instagram Reels / Shorts (9:16)": "9:16",
    "Square Social (1:1)": "1:1",
}

CAMERA_SEQUENCE = (
    "wide establishing shot with subtle forward movement",
    "stable medium tracking shot",
    "medium close-up with a gentle push-in",
    "side tracking shot with controlled parallax",
    "close-up emphasizing the current action",
    "wide continuation shot preserving screen direction",
)


@dataclass(frozen=True)
class StoryPlan:
    story: str
    beats: tuple[str, ...]
    target_seconds: int
    estimated_seconds: float
    aspect: str
    width: int
    height: int
    profile: RenderProfile

    @property
    def scene_count(self) -> int:
        return len(self.beats)


def _sentences(text: str) -> list[str]:
    cleaned = re.sub(r"\s+", " ", text.strip())
    if not cleaned:
        return []
    raw = [part.strip(" -\t") for part in re.split(r"(?<=[.!?])\s+|\n+", cleaned) if part.strip()]
    return raw or [cleaned]


def _word_chunks(text: str, target_words: int = 11) -> list[str]:
    words = text.split()
    if len(words) <= max(15, target_words + 3):
        return [text.strip()]
    chunks: list[str] = []
    cursor = 0
    while cursor < len(words):
        remaining = len(words) - cursor
        take = min(max(8, target_words), remaining)
        if remaining > target_words and remaining - take < 6:
            take = remaining
        chunk = " ".join(words[cursor: cursor + take]).strip()
        if chunk:
            chunks.append(chunk)
        cursor += take
    return chunks


def _atomic_units(story: str) -> list[str]:
    units: list[str] = []
    for sentence in _sentences(story):
        clauses = [c.strip(" ,;:-") for c in re.split(r"(?<=[,;:])\s+", sentence) if c.strip(" ,;:-")]
        if len(clauses) == 1:
            units.extend(_word_chunks(sentence))
        else:
            for clause in clauses:
                units.extend(_word_chunks(clause))
    return [u for u in units if u]


def _merge_to_count(units: list[str], count: int) -> list[str]:
    if len(units) <= count:
        return units[:]
    result: list[str] = []
    for index in range(count):
        start = round(index * len(units) / count)
        end = round((index + 1) * len(units) / count)
        result.append(" ".join(units[start:end]).strip())
    return [r for r in result if r]


def _expand_to_count(units: list[str], count: int) -> list[str]:
    work = units[:] or ["Establish the subject and begin the story clearly."]
    while len(work) < count:
        split_index = max(range(len(work)), key=lambda i: len(work[i].split()))
        words = work[split_index].split()
        if len(words) >= 10:
            midpoint = len(words) // 2
            first = " ".join(words[:midpoint]).strip()
            second = " ".join(words[midpoint:]).strip()
            work[split_index: split_index + 1] = [first, second]
            continue
        source = work[(len(work) - 1) % len(work)]
        work.append(
            "Continue naturally from the previous moment with a new visible movement or reaction while preserving this idea: "
            + source
        )
    return work[:count]


def estimate_auto_seconds(story: str) -> int:
    """Estimate watchable duration from paragraph length, capped at five minutes."""
    words = max(1, len(story.split()))
    # Rough narration/story pace: ~2.5 words/second, with an 8 second minimum.
    seconds = max(8, math.ceil(words / 2.5))
    return min(MAX_LONGFORM_SECONDS, seconds)


def plan_story(
    story: str,
    duration_label: str,
    quality_label: str,
    aspect_label: str,
) -> StoryPlan:
    if not story or not story.strip():
        raise ValueError("Enter a story or video description first.")
    profile = QUALITY_PROFILES.get(quality_label, QUALITY_PROFILES["Balanced"])
    aspect = ASPECT_LABELS.get(aspect_label, "16:9")
    width, height = profile.size_for(aspect)
    requested = DURATION_SECONDS.get(duration_label)
    target_seconds = int(requested or estimate_auto_seconds(story))
    target_seconds = max(4, min(MAX_LONGFORM_SECONDS, target_seconds))
    scene_count = max(1, math.ceil(target_seconds / profile.scene_seconds))
    scene_count = min(MAX_LONGFORM_SCENES, scene_count)

    units = _atomic_units(story)
    if len(units) > scene_count:
        beats = _merge_to_count(units, scene_count)
    else:
        beats = _expand_to_count(units, scene_count)

    estimated = len(beats) * profile.scene_seconds
    return StoryPlan(
        story=story.strip(),
        beats=tuple(beats),
        target_seconds=target_seconds,
        estimated_seconds=estimated,
        aspect=aspect,
        width=width,
        height=height,
        profile=profile,
    )


def scene_prompt(
    beat: str,
    index: int,
    total: int,
    style_prompt: str,
    character_lock: str,
) -> str:
    continuity = (
        "Opening shot: establish all important people, objects, colors and spatial relationships clearly."
        if index == 0
        else "Direct continuation of the previous shot: preserve the same identities, clothing, object shapes, colors, lighting direction, screen direction and environment."
    )
    camera = CAMERA_SEQUENCE[index % len(CAMERA_SEQUENCE)]
    character = f" Identity and character lock: {character_lock}." if character_lock.strip() else ""
    return (
        f"Scene {index + 1} of {total}. {continuity} "
        f"Chronological action: {beat}. "
        f"First show the starting pose clearly, then perform the described action smoothly, and end on a stable readable pose that can continue into the next scene."
        f"{character} Visual treatment: {style_prompt}. Camera: {camera}. "
        "Natural temporal motion, coherent anatomy and object geometry, consistent subject scale, crisp important details, no jump cut inside this shot."
    )


def plan_markdown(plan: StoryPlan, preview_limit: int = 12) -> str:
    minutes = plan.estimated_seconds / 60.0
    lines = [
        "### Automatic video plan",
        f"**{plan.scene_count} scenes** · approximately **{plan.estimated_seconds:.0f} seconds ({minutes:.1f} min)**",
        f"Native generation: **{plan.width}×{plan.height}** · {plan.profile.frames_per_scene} frames/scene · {plan.profile.label}",
        "",
    ]
    shown = min(plan.scene_count, preview_limit)
    for index in range(shown):
        lines.append(f"**Scene {index + 1}:** {plan.beats[index]}")
    if plan.scene_count > shown:
        lines.append(f"\n… plus **{plan.scene_count - shown} more automatically planned scenes**.")
    return "\n\n".join(lines)


class LongFormVideoGenerator:
    """Sequential long-form renderer designed for 6 GB-class GPUs."""

    def __init__(self, generator) -> None:
        self.generator = generator

    def generate(
        self,
        plan: StoryPlan,
        style_prompt: str,
        character_lock: str,
        reference_image: Image.Image | None,
        negative_prompt: str,
        seed: int = -1,
        progress_callback: Progress = None,
    ) -> Path:
        clips: list[Path] = []
        previous_frame = reference_image
        total = plan.scene_count

        for index, beat in enumerate(plan.beats):
            prompt = scene_prompt(beat, index, total, style_prompt, character_lock)
            scene_seed = -1 if seed is None or int(seed) < 0 else int(seed) + index * 17

            def scene_progress(message: str, value: float, idx: int = index) -> None:
                if progress_callback:
                    local = min(1.0, max(0.0, float(value)))
                    progress_callback(
                        f"Scene {idx + 1}/{total}: {message}",
                        (idx + local) / total,
                    )

            scene_progress("preparing", 0.02)
            if previous_frame is None:
                clip = self.generator.generate_text_to_video(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    width=plan.width,
                    height=plan.height,
                    num_frames=plan.profile.frames_per_scene,
                    num_inference_steps=plan.profile.inference_steps,
                    guidance_scale=plan.profile.guidance_scale,
                    seed=scene_seed,
                    progress_callback=scene_progress,
                    character_lock=character_lock,
                )
            else:
                clip = self.generator.generate_image_to_video(
                    prompt=prompt,
                    image=previous_frame,
                    negative_prompt=negative_prompt,
                    width=plan.width,
                    height=plan.height,
                    num_frames=plan.profile.frames_per_scene,
                    num_inference_steps=plan.profile.inference_steps,
                    guidance_scale=plan.profile.guidance_scale,
                    seed=scene_seed,
                    progress_callback=scene_progress,
                    character_lock=character_lock,
                )

            clip_path = Path(clip)
            clips.append(clip_path)
            previous_frame = extract_last_frame(clip_path)
            scene_progress("complete", 1.0)

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output = OUTPUTS_DIR / f"longform_{stamp}.mp4"
        if progress_callback:
            progress_callback("Joining scenes into the final video", 0.995)
        concatenate_videos_streaming(clips, output, target_fps=DEFAULT_FPS)
        if progress_callback:
            progress_callback("Long-form video complete", 1.0)
        return output
