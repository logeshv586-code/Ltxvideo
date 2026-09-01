"""Automatic long-form scene planning and low-VRAM sequential generation.

The customer supplies one paragraph. This module turns it into chronological
shots and generates them one at a time. Continuation shots are conditioned on
a short *sequence* of previous frames rather than one terminal still frame.
That preserves motion direction and avoids the recursive blur/geometry drift
seen in the earlier last-frame I2V implementation.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

from PIL import Image

from config import OUTPUTS_DIR
from engine.video_processor import (
    concatenate_videos_streaming,
    extract_tail_frames,
    trim_video_start_frames,
)

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
    fps: int = 24
    tail_frames: int = 17
    scene_retries: int = 1

    @property
    def scene_seconds(self) -> float:
        return self.frames_per_scene / self.fps

    def size_for(self, aspect: str) -> tuple[int, int]:
        if aspect == "9:16":
            return self.portrait
        if aspect == "1:1":
            return self.square
        return self.landscape

    @property
    def continuation_overlap(self) -> int:
        # 17 conditioning frames -> 16 overlap frames. The overlap remains a
        # multiple of 8 so adding it to an 8k+1 LTX clip stays 8k+1.
        return max(0, self.tail_frames - 1)


QUALITY_PROFILES: dict[str, RenderProfile] = {
    "Fast": RenderProfile(
        key="fast",
        label="Fast • draft/long-form • 24 fps",
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
        label="Balanced • recommended safe RTX 4050 6 GB mode • 24 fps",
        landscape=(512, 288),
        portrait=(288, 512),
        square=(384, 384),
        frames_per_scene=97,
        inference_steps=16,
        guidance_scale=3.0,
        delivery_long_edge=1280,
    ),
    "Reference 720p": RenderProfile(
        key="reference720",
        label="Reference look • 576×320 native → clean 720p delivery • 24 fps",
        landscape=(576, 320),
        portrait=(320, 576),
        square=(448, 448),
        frames_per_scene=97,
        inference_steps=18,
        guidance_scale=3.2,
        delivery_long_edge=1280,
    ),
    "Quality": RenderProfile(
        key="quality",
        label="Quality • more native detail, slower • 24 fps",
        landscape=(576, 320),
        portrait=(320, 576),
        square=(448, 448),
        frames_per_scene=121,
        inference_steps=20,
        guidance_scale=3.2,
        delivery_long_edge=1280,
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

CONTINUOUS_CAMERA_SEQUENCE = (
    "stable cinematic tracking with the same screen direction",
    "gentle forward tracking while preserving the same subject scale",
    "subtle lateral tracking with continuous camera momentum",
)

SCENE_CHANGE_HINTS = re.compile(
    r"\b(then|afterwards?|later|meanwhile|next|suddenly|cut to|cuts to|"
    r"arrives?|enters?|leaves?|switches?|changes? to|another scene|finally)\b",
    flags=re.IGNORECASE,
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
    continuity_mode: str

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


def _continuity_mode(story: str) -> str:
    """Short single-action prompts become extensions instead of fake scene cuts."""
    sentences = _sentences(story)
    words = story.split()
    if len(words) <= 45 and len(sentences) <= 2 and not SCENE_CHANGE_HINTS.search(story):
        return "continuous"
    return "storyboard"


def estimate_auto_seconds(story: str) -> int:
    """Estimate watchable duration from paragraph length, capped at five minutes."""
    words = max(1, len(story.split()))
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

    mode = _continuity_mode(story)
    if mode == "continuous":
        # Keep the semantic subject/action constant. The continuation pipeline
        # supplies actual temporal evolution rather than prompting fresh scenes.
        beats = [story.strip()] * scene_count
    else:
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
        continuity_mode=mode,
    )


def scene_prompt(
    beat: str,
    index: int,
    total: int,
    style_prompt: str,
    character_lock: str,
    continuity_mode: str = "storyboard",
) -> str:
    if continuity_mode == "continuous":
        continuity = (
            "Opening shot: establish the subject, environment, colors and screen direction clearly."
            if index == 0
            else "Seamless extension of the same moving shot: do not restart the action, do not replace the subject, and preserve motion direction, camera momentum, identity, colors, lighting and geometry from the conditioned video tail."
        )
        camera = CONTINUOUS_CAMERA_SEQUENCE[index % len(CONTINUOUS_CAMERA_SEQUENCE)]
        action = (
            f"Chronological action: {beat}."
            if index == 0
            else f"Continue the same chronological action naturally without resetting: {beat}."
        )
    else:
        continuity = (
            "Opening shot: establish all important people, objects, colors and spatial relationships clearly."
            if index == 0
            else "Direct continuation of the previous story moment: preserve the same identities, clothing, object shapes, colors, lighting direction, screen direction and environment unless the story explicitly changes them."
        )
        camera = CAMERA_SEQUENCE[index % len(CAMERA_SEQUENCE)]
        action = f"Chronological action: {beat}."

    character = f" Identity and character lock: {character_lock}." if character_lock.strip() else ""
    return (
        f"Scene {index + 1} of {total}. {continuity} "
        f"{action} "
        "Show readable poses and smooth purposeful motion. Preserve clean subject geometry and stable proportions throughout the shot."
        f"{character} Visual treatment: {style_prompt}. Camera: {camera}. "
        "Premium coherent animation, natural temporal motion, stable faces and anatomy, clean object geometry, crisp important details, no morphing, no flicker, no duplicate subject, no jump cut inside this shot."
    )


def plan_markdown(plan: StoryPlan, preview_limit: int = 12) -> str:
    minutes = plan.estimated_seconds / 60.0
    mode_label = "continuous motion extension" if plan.continuity_mode == "continuous" else "storyboard scenes"
    lines = [
        "### Automatic video plan",
        f"**{plan.scene_count} shots** · approximately **{plan.estimated_seconds:.0f} seconds ({minutes:.1f} min)** · **{mode_label}**",
        (
            f"Native generation: **{plan.width}×{plan.height}** · "
            f"{plan.profile.frames_per_scene} new frames/shot · **{plan.profile.fps} fps** · {plan.profile.label}"
        ),
        f"Continuity: **{plan.profile.tail_frames}-frame motion tail** between related shots; failed visual tails retry before chaining.",
        "",
    ]
    shown = min(plan.scene_count, preview_limit)
    for index in range(shown):
        lines.append(f"**Shot {index + 1}:** {plan.beats[index]}")
    if plan.scene_count > shown:
        lines.append(f"\n… plus **{plan.scene_count - shown} more automatically planned shots**.")
    return "\n\n".join(lines)


def _aligned_tail(frames: list[Image.Image], preferred: int) -> list[Image.Image]:
    """Choose a tail length whose overlap is divisible by eight."""
    if not frames:
        return []
    available = min(len(frames), max(1, int(preferred)))
    valid = 1 + 8 * max(0, (available - 1) // 8)
    return frames[-valid:]


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
        previous_tail: list[Image.Image] = []
        total = plan.scene_count

        if not hasattr(self.generator, "generate_conditioned_video"):
            raise RuntimeError(
                "This long-form build requires the multi-frame LTX condition pipeline. "
                "Update the generator before rendering."
            )

        for index, beat in enumerate(plan.beats):
            prompt = scene_prompt(
                beat,
                index,
                total,
                style_prompt,
                character_lock,
                continuity_mode=plan.continuity_mode,
            )
            base_seed = -1 if seed is None or int(seed) < 0 else int(seed) + index * 17
            tail = _aligned_tail(previous_tail, plan.profile.tail_frames)
            overlap_frames = max(0, len(tail) - 1)
            render_frames = plan.profile.frames_per_scene + overlap_frames

            def scene_progress(message: str, value: float, idx: int = index) -> None:
                if progress_callback:
                    local = min(1.0, max(0.0, float(value)))
                    progress_callback(
                        f"Shot {idx + 1}/{total}: {message}",
                        (idx + local) / total,
                    )

            accepted: Path | None = None
            attempts = 1 + max(0, int(plan.profile.scene_retries))
            for attempt in range(attempts):
                attempt_seed = base_seed
                if attempt and base_seed >= 0:
                    attempt_seed = base_seed + 100_003 * attempt

                if attempt:
                    scene_progress("visual QC requested a safer retry", 0.06)

                raw_clip = self.generator.generate_conditioned_video(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    width=plan.width,
                    height=plan.height,
                    num_frames=render_frames,
                    num_inference_steps=plan.profile.inference_steps,
                    guidance_scale=plan.profile.guidance_scale,
                    seed=attempt_seed,
                    progress_callback=scene_progress,
                    character_lock=character_lock,
                    fps=plan.profile.fps,
                    conditioning_frames=tail or None,
                    reference_image=reference_image if index == 0 and not tail else None,
                    condition_strength=1.0,
                    image_cond_noise_scale=0.05 if attempt == 0 else 0.0,
                )

                raw_path = Path(raw_clip)
                qc = self.generator.last_qc
                failed = bool(qc and (qc.fatal or qc.visual_failure))
                if failed and attempt + 1 < attempts:
                    continue
                if failed:
                    details = qc.summary() if qc else "Unknown scene quality failure."
                    raise RuntimeError(
                        f"Shot {index + 1} failed visual QC twice. The renderer stopped instead of "
                        f"feeding a damaged tail into later shots.\n{details}"
                    )

                if overlap_frames:
                    segment = raw_path.with_name(raw_path.stem + "_new.mp4")
                    trim_video_start_frames(
                        raw_path,
                        segment,
                        frames_to_trim=overlap_frames,
                        target_fps=plan.profile.fps,
                    )
                    accepted = segment
                else:
                    accepted = raw_path
                break

            if accepted is None:
                raise RuntimeError(f"Shot {index + 1} did not produce an accepted clip.")

            clips.append(accepted)
            previous_tail = extract_tail_frames(
                accepted,
                frame_count=plan.profile.tail_frames,
                safety_margin=1,
            )
            scene_progress("complete; motion tail captured", 1.0)

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output = OUTPUTS_DIR / f"longform_{stamp}.mp4"
        if progress_callback:
            progress_callback("Joining accepted shots into the final video", 0.995)
        concatenate_videos_streaming(clips, output, target_fps=plan.profile.fps)
        if progress_callback:
            progress_callback("Long-form video complete", 1.0)
        return output
