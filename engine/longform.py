"""Personal video planning and low-VRAM sequential rendering.

The user describes the video once. The planner understands the requested total
length and turns it into GPU-sized clips. RTX 4050 users can render one 4-second
or 8-second clip directly; longer videos are generated clip-by-clip and joined
while keeping story context and motion continuity.
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
from engine.video_processor import concatenate_videos_streaming, extract_tail_frames, trim_video_start_frames

Progress = Callable[[str, float], None] | None
MAX_LONGFORM_SECONDS = 300
MAX_LONGFORM_SCENES = 96

GENERATION_MODES = ("Single Clip", "Continuous Video")
CLIP_LENGTHS = {
    "4 seconds • Recommended": 97,
    "8 seconds • Max practical": 193,
}


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


QUALITY_PROFILES: dict[str, RenderProfile] = {
    "Fast": RenderProfile(
        key="fast",
        label="Fast • fastest 720p delivery",
        landscape=(384, 224),
        portrait=(224, 384),
        square=(320, 320),
        frames_per_scene=97,
        inference_steps=12,
        guidance_scale=3.5,
        delivery_long_edge=1280,
    ),
    "Balanced": RenderProfile(
        key="balanced",
        label="Balanced • recommended for RTX 4050 6 GB",
        landscape=(512, 288),
        portrait=(288, 512),
        square=(384, 384),
        frames_per_scene=97,
        inference_steps=16,
        guidance_scale=4.0,
        delivery_long_edge=1280,
    ),
    "High": RenderProfile(
        key="high",
        label="High • best detail; 4-second clips recommended",
        landscape=(576, 320),
        portrait=(320, 576),
        square=(448, 448),
        frames_per_scene=97,
        inference_steps=20,
        guidance_scale=4.0,
        delivery_long_edge=1280,
    ),
}

DURATION_SECONDS = {
    "15 seconds": 15,
    "30 seconds": 30,
    "1 minute": 60,
    "2 minutes": 120,
    "3 minutes": 180,
    "4 minutes": 240,
    "5 minutes": 300,
}

ASPECT_LABELS = {
    "Landscape • 1280×720 • 16:9": "16:9",
    "Portrait • 720×1280 • 9:16": "9:16",
    "Square • 720×720 • 1:1": "1:1",
}


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
    generation_mode: str = "Continuous Video"
    clip_frames: int = 97

    @property
    def scene_count(self) -> int:
        return len(self.beats)

    @property
    def clip_seconds(self) -> float:
        return self.clip_frames / self.profile.fps


ACTION_SPLIT = re.compile(
    r"\s*(?:\.|!|\?|;|\n|,\s*then\s+|\bthen\b|\band then\b|\bafter that\b|\bafterwards\b|\bnext\b|\bfinally\b)\s*",
    flags=re.IGNORECASE,
)


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def _action_units(story: str) -> list[str]:
    text = _clean(story)
    if not text:
        return []
    parts = [p.strip(" ,.-") for p in ACTION_SPLIT.split(text) if p.strip(" ,.-")]
    if len(parts) <= 1 and " and " in text.lower():
        rough = re.split(r"\s+and\s+", text, flags=re.IGNORECASE)
        parts = [p.strip(" ,.-") for p in rough if p.strip(" ,.-")]
    return parts or [text]


def _merge_units(units: list[str], count: int) -> list[str]:
    if not units:
        return ["Establish the requested subject and action clearly."] * count
    if len(units) == count:
        return units
    if len(units) > count:
        out: list[str] = []
        for index in range(count):
            start = round(index * len(units) / count)
            end = round((index + 1) * len(units) / count)
            out.append("; then ".join(units[start:end]))
        return out

    out = units[:]
    while len(out) < count:
        source = out[-1]
        phase = len(out) + 1
        out.append(
            f"Continue naturally from the previous moment. Progress the same story and action further without restarting it; continuation phase {phase}: {source}"
        )
    return out[:count]


def estimate_auto_seconds(story: str) -> int:
    words = max(1, len(story.split()))
    return min(MAX_LONGFORM_SECONDS, max(8, math.ceil(words / 2.5)))


def plan_story(
    story: str,
    duration_label: str,
    quality_label: str,
    aspect_label: str,
    generation_mode: str = "Continuous Video",
    clip_length_label: str = "4 seconds • Recommended",
) -> StoryPlan:
    """Create a customer-facing plan while keeping GPU chunking internal."""
    story = _clean(story)
    if not story:
        raise ValueError("Describe what you want the video to show.")

    profile = QUALITY_PROFILES.get(quality_label, QUALITY_PROFILES["Balanced"])
    aspect = ASPECT_LABELS.get(aspect_label, "16:9")
    width, height = profile.size_for(aspect)
    clip_frames = CLIP_LENGTHS.get(clip_length_label, 97)
    clip_seconds = clip_frames / profile.fps
    mode = generation_mode if generation_mode in GENERATION_MODES else "Continuous Video"

    if mode == "Single Clip":
        target_seconds = int(round(clip_seconds))
        beats = [story]
        continuity_mode = "single"
    else:
        target_seconds = int(DURATION_SECONDS.get(duration_label, 15))
        target_seconds = max(15, min(MAX_LONGFORM_SECONDS, target_seconds))
        count = min(MAX_LONGFORM_SCENES, max(1, math.ceil(target_seconds / clip_seconds)))
        beats = _merge_units(_action_units(story), count)
        continuity_mode = "continuous"

    return StoryPlan(
        story=story,
        beats=tuple(beats),
        target_seconds=target_seconds,
        estimated_seconds=len(beats) * clip_seconds,
        aspect=aspect,
        width=width,
        height=height,
        profile=profile,
        continuity_mode=continuity_mode,
        generation_mode=mode,
        clip_frames=clip_frames,
    )


def scene_prompt(
    beat: str,
    index: int,
    total: int,
    style_prompt: str,
    character_lock: str,
    continuity_mode: str = "continuous",
    full_story: str = "",
) -> str:
    opening = index == 0
    continuity = (
        "Opening clip: establish all important subjects, their appearance, the environment and the intended action clearly."
        if opening
        else "Continue directly from the conditioned previous motion. Do not restart the story, replace the subject, change identity, teleport objects or reset camera direction."
    )
    character = f" Character/subject lock: {character_lock}." if character_lock.strip() else ""
    context = f" Overall requested video: {full_story}." if full_story else ""
    return (
        f"Video clip {index + 1} of {total}. {continuity}{context} "
        f"Current timeline action: {beat}. "
        "Show a clear beginning, visible progression and readable end pose for this clip while the overall video continues forward. "
        f"{character} Visual style: {style_prompt}. "
        "Smooth purposeful motion, stable subject identity, coherent anatomy and object geometry, consistent colors and lighting, "
        "cinematic composition, clean detail, no morphing, no duplicate subjects, no flicker, no sudden jump cut."
    )


def plan_markdown(plan: StoryPlan, preview_limit: int = 8) -> str:
    if plan.aspect == "9:16":
        delivery = "720×1280"
    elif plan.aspect == "1:1":
        delivery = "720×720"
    else:
        delivery = "1280×720"

    if plan.generation_mode == "Single Clip":
        headline = f"**Single clip · {plan.clip_seconds:.1f} sec**"
    else:
        headline = f"**Continuous video · {plan.target_seconds} sec**"

    lines = [
        "### Video setup",
        headline,
        f"**Quality:** {plan.profile.label}",
        f"**Output:** {delivery} · {plan.profile.fps} fps · {plan.aspect}",
        f"**RTX 4050 render chunks:** {plan.scene_count} × about {plan.clip_seconds:.1f} sec",
    ]
    if plan.generation_mode != "Single Clip":
        lines.append("The backend continues each clip from the previous clip automatically and trims the final delivery to the selected duration.")
    lines.append("")
    for i, beat in enumerate(plan.beats[:preview_limit]):
        lines.append(f"**Part {i + 1}:** {beat}")
    if len(plan.beats) > preview_limit:
        lines.append(f"… plus **{len(plan.beats) - preview_limit} more continuation parts**.")
    return "\n\n".join(lines)


def _aligned_tail(frames: list[Image.Image], preferred: int) -> list[Image.Image]:
    if not frames:
        return []
    available = min(len(frames), max(1, int(preferred)))
    valid = 1 + 8 * max(0, (available - 1) // 8)
    return frames[-valid:]


class LongFormVideoGenerator:
    """Generate one clip or a continuous multi-clip video on a 6 GB GPU."""

    def __init__(self, generator) -> None:
        self.generator = generator

    def _first_clip(
        self,
        prompt: str,
        plan: StoryPlan,
        reference_image: Image.Image | None,
        negative_prompt: str,
        character_lock: str,
        seed: int,
        callback: Progress,
    ) -> Path:
        kwargs = dict(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=plan.width,
            height=plan.height,
            num_frames=plan.clip_frames,
            num_inference_steps=plan.profile.inference_steps,
            guidance_scale=plan.profile.guidance_scale,
            seed=seed,
            progress_callback=callback,
            character_lock=character_lock,
            fps=plan.profile.fps,
        )
        if reference_image is not None:
            return Path(self.generator.generate_image_to_video(image=reference_image, **kwargs))
        return Path(self.generator.generate_text_to_video(**kwargs))

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

        for index, beat in enumerate(plan.beats):
            prompt = scene_prompt(
                beat,
                index,
                total,
                style_prompt,
                character_lock,
                continuity_mode=plan.continuity_mode,
                full_story=plan.story,
            )
            clip_seed = -1 if seed is None or int(seed) < 0 else int(seed) + index * 17

            def local_progress(message: str, value: float, idx: int = index) -> None:
                if progress_callback:
                    progress_callback(
                        f"Part {idx + 1}/{total}: {message}",
                        (idx + min(1.0, max(0.0, float(value)))) / total,
                    )

            if index == 0:
                accepted = self._first_clip(
                    prompt,
                    plan,
                    reference_image,
                    negative_prompt,
                    character_lock,
                    clip_seed,
                    local_progress,
                )
            else:
                tail = _aligned_tail(previous_tail, plan.profile.tail_frames)
                overlap = max(0, len(tail) - 1)
                render_frames = plan.clip_frames + overlap
                accepted = None

                # Preferred path: official LTX multi-frame conditioning.
                try:
                    raw = Path(
                        self.generator.generate_conditioned_video(
                            prompt=prompt,
                            negative_prompt=negative_prompt,
                            width=plan.width,
                            height=plan.height,
                            num_frames=render_frames,
                            num_inference_steps=plan.profile.inference_steps,
                            guidance_scale=plan.profile.guidance_scale,
                            seed=clip_seed,
                            progress_callback=local_progress,
                            character_lock=character_lock,
                            fps=plan.profile.fps,
                            conditioning_frames=tail,
                            condition_strength=1.0,
                            image_cond_noise_scale=0.025,
                        )
                    )
                    if overlap:
                        accepted = raw.with_name(raw.stem + "_new.mp4")
                        trim_video_start_frames(raw, accepted, overlap, target_fps=plan.profile.fps)
                    else:
                        accepted = raw
                except Exception as exc:
                    # Compatibility fallback: never leave the customer with a
                    # generic red Gradio error if condition mode is unavailable.
                    # Use a stable frame from the motion tail to continue via I2V.
                    local_progress(f"Multi-frame continuation fallback: {type(exc).__name__}", 0.08)
                    if not tail:
                        raise RuntimeError(f"Continuation failed: {exc}") from exc
                    anchor = tail[max(0, len(tail) - 5)]
                    accepted = Path(
                        self.generator.generate_image_to_video(
                            prompt=prompt,
                            image=anchor,
                            negative_prompt=negative_prompt,
                            width=plan.width,
                            height=plan.height,
                            num_frames=plan.clip_frames,
                            num_inference_steps=plan.profile.inference_steps,
                            guidance_scale=plan.profile.guidance_scale,
                            seed=clip_seed,
                            progress_callback=local_progress,
                            character_lock=character_lock,
                            fps=plan.profile.fps,
                        )
                    )

            qc = self.generator.last_qc
            if qc and qc.fatal:
                raise RuntimeError(f"Part {index + 1} failed technical QC.\n{qc.summary()}")
            clips.append(Path(accepted))
            if plan.generation_mode != "Single Clip" and index + 1 < total:
                previous_tail = extract_tail_frames(
                    accepted,
                    frame_count=plan.profile.tail_frames,
                    safety_margin=1,
                )
            local_progress("complete", 1.0)

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output = OUTPUTS_DIR / f"video_{stamp}.mp4"
        if progress_callback:
            progress_callback("Joining generated parts", 0.995)
        concatenate_videos_streaming(clips, output, target_fps=plan.profile.fps)
        if progress_callback:
            progress_callback("Video generation complete", 1.0)
        return output
