"""Personal video planning and low-VRAM sequential rendering.

The user describes the video once. The planner turns that request into a
chronological timeline and renders it in GPU-sized clips. RTX 4050 users can
render one 4-second or 8-second clip directly; longer videos are generated
clip-by-clip and joined while preserving story context and motion continuity.

Quality v4 treats low-resolution Fast output as draft-only. Customer-facing
final renders use Balanced or High native generation, strip prompt metadata
from the visible action, and keep each short diffusion clip focused on only the
amount of action that can realistically happen in that clip.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Callable

import torch
from PIL import Image

from config import OUTPUTS_DIR
from engine.generator import is_fatal_cuda_error
from engine.video_processor import concatenate_videos_streaming, extract_tail_frames, trim_video_start_frames

Progress = Callable[[str, float], None] | None
MAX_LONGFORM_SECONDS = 600
MAX_LONGFORM_SCENES = 150

GENERATION_MODES = ("Single Clip", "Continuous Video")
CLIP_LENGTHS = {
    "4 seconds • Recommended": 97,
    "8 seconds • Max practical": 193,
    "10 seconds • 2xT4 High Memory": 241,
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
    # Kept for legacy/developer use. The personal-video-maker UI no longer
    # presents this as a normal final-quality choice.
    "Fast": RenderProfile(
        key="fast",
        label="Draft only • 384×224 native • fastest preview",
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
        label="Balanced • 512×288 native • RTX 4050 safe final",
        landscape=(512, 288),
        portrait=(288, 512),
        square=(384, 384),
        frames_per_scene=97,
        inference_steps=20,
        guidance_scale=4.0,
        delivery_long_edge=1280,
    ),
    "High": RenderProfile(
        key="high",
        label="High • 576×320 native • best 4-second detail",
        landscape=(576, 320),
        portrait=(320, 576),
        square=(448, 448),
        frames_per_scene=97,
        # A short T4 clip can afford a little more denoising than the old
        # 24-step preset. This reduces unfinished texture and edge noise before
        # export without claiming to add detail beyond the model's native frame.
        inference_steps=28,
        guidance_scale=4.75,
        delivery_long_edge=1280,
    ),
}

# The main customer UI intentionally hides the low-resolution draft profile.
CUSTOMER_QUALITY_CHOICES = ("Balanced", "High")

DURATION_SECONDS = {
    "15 seconds": 15,
    "30 seconds": 30,
    "1 minute": 60,
    "2 minutes": 120,
    "3 minutes": 180,
    "4 minutes": 240,
    "5 minutes": 300,
    "10 minutes": 600,
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
    style_hint: str = ""
    camera_hint: str = ""

    @property
    def scene_count(self) -> int:
        return len(self.beats)

    @property
    def clip_seconds(self) -> float:
        return self.clip_frames / self.profile.fps


ACTION_SPLIT = re.compile(
    r"\s*(?:\.|!|\?|;|\n|,\s*then\s+|\band then\b|\bafter that\b|\bafterwards\b|\bfinally\b|\bnext\b|\bthen\b)\s*",
    flags=re.IGNORECASE,
)

DIRECTIVE_LINE = re.compile(
    r"^\s*(style|look|duration|camera|aspect(?:\s+ratio)?|resolution|size|quality|fps|frame\s*rate)\s*:\s*(.*?)\s*$",
    flags=re.IGNORECASE,
)

VISUAL_ACTION_HINT = re.compile(
    r"\b(sit|sits|stand|stands|walk|walks|run|runs|jump|jumps|move|moves|fly|flies|float|floats|"
    r"look|looks|smile|smiles|turn|turns|open|opens|close|closes|pick|picks|hold|holds|raise|raises|"
    r"wave|waves|cook|cooks|make|makes|help|helps|speak|speaks|talk|talks|appear|appears|enter|enters|"
    r"leave|leaves|dance|dances|play|plays|reach|reaches|eat|eats|drink|drinks|drive|drives|cruise|cruises)\b",
    flags=re.IGNORECASE,
)


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def _shorten_words(text: str, max_words: int) -> str:
    words = _clean(text).split()
    if len(words) <= max_words:
        return " ".join(words)
    return " ".join(words[:max_words]).rstrip(" ,;:-")


def extract_prompt_directives(text: str) -> tuple[str, dict[str, str]]:
    """Separate prompt metadata from visible story/action content.

    Customers often paste prompt-engineering notes such as ``Style:``,
    ``Duration:`` or ``Camera:``. Sending those lines verbatim into every LTX
    shot can contradict the UI selection and overload a four-second clip. The
    planner therefore keeps useful style/camera hints separately and removes
    timing/format metadata from the visible timeline.
    """
    directives: dict[str, str] = {}
    kept: list[str] = []
    for line in (text or "").splitlines():
        match = DIRECTIVE_LINE.match(line)
        if not match:
            kept.append(line)
            continue
        key = match.group(1).lower().replace(" ", "")
        value = _clean(match.group(2))
        if value:
            directives[key] = value
    return _clean(" ".join(kept)), directives


def _action_units(story: str, target_count: int = 1) -> list[str]:
    """Extract chronological actions even from casual run-on user prompts."""
    text = _clean(story)
    if not text:
        return []
    parts = [p.strip(" ,.-") for p in ACTION_SPLIT.split(text) if p.strip(" ,.-")]

    # Users often type one long sentence with repeated "and" instead of
    # punctuation. If strong transitions did not produce enough beats, use
    # conjunctions as secondary boundaries.
    if len(parts) < target_count and " and " in text.lower():
        expanded: list[str] = []
        for part in parts or [text]:
            if len(part.split()) >= 8 and " and " in part.lower():
                sub = [
                    p.strip(" ,.-")
                    for p in re.split(r"\s+and\s+", part, flags=re.IGNORECASE)
                    if p.strip(" ,.-")
                ]
                expanded.extend(sub or [part])
            else:
                expanded.append(part)
        if len(expanded) > len(parts):
            parts = expanded
    return parts or [text]


def _single_clip_action(story: str, max_words: int = 44) -> str:
    """Choose only the first visually achievable action for one 4/8s clip."""
    units = _action_units(story, target_count=2)
    if not units:
        return "Establish the requested subject and perform one clear visible action."

    visual = [unit for unit in units if VISUAL_ACTION_HINT.search(unit)]
    chosen = (visual or units)[:2]
    return _shorten_words("; then ".join(chosen), max_words)


def _merge_units(units: list[str], count: int) -> list[str]:
    if not units:
        return ["Establish the requested subject and action clearly."] * count
    if len(units) == count:
        return [_shorten_words(unit, 44) for unit in units]
    if len(units) > count:
        out: list[str] = []
        for index in range(count):
            start = round(index * len(units) / count)
            end = round((index + 1) * len(units) / count)
            out.append(_shorten_words("; then ".join(units[start:end]), 44))
        return out

    out = [_shorten_words(unit, 44) for unit in units]
    while len(out) < count:
        source = out[-1]
        phase = len(out) + 1
        out.append(
            _shorten_words(
                f"Continue the previous action naturally without restarting it; progression phase {phase}: {source}",
                44,
            )
        )
    return out[:count]


def estimate_auto_seconds(story: str) -> int:
    words = max(1, len(story.split()))
    return min(MAX_LONGFORM_SECONDS, max(8, math.ceil(words / 2.5)))


def _memory_safe_profile(profile: RenderProfile, clip_frames: int) -> RenderProfile:
    """Keep High 8-second clips practical on 6 GB VRAM unless we have multi-GPU."""
    if profile.key != "high" or clip_frames <= 97:
        return profile
    
    # If we have multiple GPUs (e.g., dual T4), we have plenty of memory to skip downscaling
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        return profile

    return replace(
        profile,
        label="High • 8-second memory-safe • 512×288 native",
        landscape=(512, 288),
        portrait=(288, 512),
        square=(384, 384),
        inference_steps=22,
        guidance_scale=4.25,
    )


def plan_story(
    story: str,
    duration_label: str,
    quality_label: str,
    aspect_label: str,
    generation_mode: str = "Continuous Video",
    clip_length_label: str = "4 seconds • Recommended",
) -> StoryPlan:
    clean_story, directives = extract_prompt_directives(story)
    if not clean_story:
        raise ValueError("Describe what you want the video to show.")

    requested_profile = QUALITY_PROFILES.get(quality_label, QUALITY_PROFILES["Balanced"])
    clip_frames = CLIP_LENGTHS.get(clip_length_label, 97)
    profile = _memory_safe_profile(requested_profile, clip_frames)
    aspect = ASPECT_LABELS.get(aspect_label, "16:9")
    width, height = profile.size_for(aspect)
    clip_seconds = clip_frames / profile.fps
    mode = generation_mode if generation_mode in GENERATION_MODES else "Continuous Video"

    if mode == "Single Clip":
        target_seconds = int(round(clip_seconds))
        beats = [_single_clip_action(clean_story, 44 if clip_frames <= 97 else 64)]
        continuity_mode = "single"
    else:
        target_seconds = int(DURATION_SECONDS.get(duration_label, 15))
        target_seconds = max(15, min(MAX_LONGFORM_SECONDS, target_seconds))
        count = min(MAX_LONGFORM_SCENES, max(1, math.ceil(target_seconds / clip_seconds)))
        beats = _merge_units(_action_units(clean_story, target_count=count), count)
        continuity_mode = "continuous"

    style_hint = directives.get("style", directives.get("look", ""))
    camera_hint = directives.get("camera", "")
    return StoryPlan(
        story=clean_story,
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
        style_hint=style_hint,
        camera_hint=camera_hint,
    )


def scene_prompt(
    beat: str,
    index: int,
    total: int,
    style_prompt: str,
    character_lock: str,
    continuity_mode: str = "continuous",
    full_story: str = "",
    camera_hint: str = "",
) -> str:
    if index == 0:
        continuity = "Establish the important subject, environment and spatial relationships clearly in one coherent shot."
    else:
        continuity = (
            "Continue directly from the conditioned previous motion. Preserve subject identity, proportions, colors, "
            "lighting, screen direction and camera momentum; do not restart the story or teleport objects."
        )

    if continuity_mode == "single":
        focus = "This is one short clip: perform only the current visible action and do not rush unrelated later events into this shot."
    else:
        focus = "Advance only the current timeline beat; leave later events for later continuation clips."

    character = (
        f" Subject lock: {_shorten_words(character_lock, 36)}."
        if character_lock.strip()
        else ""
    )
    # ``full_story`` often includes a whole character and environment bible.
    # The lock and current beat carry the useful information without exceeding
    # the LTX text encoder's 128-token context.
    context = ""
    camera = (
        f" Camera direction: {_shorten_words(camera_hint, 24)}."
        if camera_hint.strip()
        else ""
    )
    action = _shorten_words(beat, 48)

    return (
        f"Video clip {index + 1} of {total}. {continuity}{context} "
        f"Current visible action: {action}. {focus}{character}{camera} "
        f"Visual style: {style_prompt}. "
        "Clean stable anatomy, coherent materials, controlled lighting, no morphing, duplicates, flicker, text or identity change."
    )


def plan_markdown(plan: StoryPlan, preview_limit: int = 8) -> str:
    delivery = "720×1280" if plan.aspect == "9:16" else ("720×720" if plan.aspect == "1:1" else "1280×720")
    headline = (
        f"**Single clip · {plan.clip_seconds:.1f} sec**"
        if plan.generation_mode == "Single Clip"
        else f"**Continuous video · {plan.target_seconds} sec**"
    )
    lines = [
        "### Video setup",
        headline,
        f"**Quality:** {plan.profile.label}",
        f"**Native generation:** {plan.width}×{plan.height} · {plan.profile.inference_steps} steps",
        f"**Output:** {delivery} · {plan.profile.fps} fps · {plan.aspect}",
        f"**RTX 4050 render parts:** {plan.scene_count} × about {plan.clip_seconds:.1f} sec",
    ]
    if plan.style_hint:
        lines.append(f"**Detected style instruction:** {plan.style_hint}")
    if plan.camera_hint:
        lines.append(f"**Detected camera instruction:** {plan.camera_hint}")
    if plan.generation_mode != "Single Clip":
        lines.append("Each part continues from the previous one automatically; the final file is trimmed to the selected duration.")
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


def _raise_if_fatal_cuda_error(exc: BaseException) -> None:
    """Do not conceal a poisoned CUDA context behind a continuation fallback."""
    if is_fatal_cuda_error(exc):
        raise RuntimeError(
            "CUDA reported illegal-memory-access (error 700). The CUDA context is unsafe, so the job stopped "
            "before trying another fallback. Restart the server, then rerun the story; completed part files "
            "remain in outputs/."
        ) from exc


class LongFormVideoGenerator:
    """Generate one clip or a continuous multi-clip video on a 6 GB GPU."""

    def __init__(self, generator) -> None:
        self.generator = generator

    def _base_kwargs(
        self,
        prompt: str,
        plan: StoryPlan,
        negative_prompt: str,
        character_lock: str,
        seed: int,
        callback: Progress,
        frames: int | None = None,
    ) -> dict:
        return dict(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=plan.width,
            height=plan.height,
            num_frames=int(frames or plan.clip_frames),
            num_inference_steps=plan.profile.inference_steps,
            guidance_scale=plan.profile.guidance_scale,
            seed=seed,
            progress_callback=callback,
            character_lock=character_lock,
            fps=plan.profile.fps,
            # Connected clips must use one stable native size. This prevents a
            # text-only T4 upscale from being applied to heavier conditioning.
            _adaptive_native_upscale=False,
        )

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
        kwargs = self._base_kwargs(prompt, plan, negative_prompt, character_lock, seed, callback)
        try:
            if reference_image is not None:
                return Path(self.generator.generate_image_to_video(image=reference_image, **kwargs))
            return Path(self.generator.generate_text_to_video(**kwargs))
        except Exception as exc:
            _raise_if_fatal_cuda_error(exc)
            raise

    def _i2v_fallback(
        self,
        prompt: str,
        plan: StoryPlan,
        anchor: Image.Image,
        negative_prompt: str,
        character_lock: str,
        seed: int,
        callback: Progress,
    ) -> Path:
        try:
            return Path(
                self.generator.generate_image_to_video(
                    image=anchor,
                    **self._base_kwargs(prompt, plan, negative_prompt, character_lock, seed, callback),
                )
            )
        except Exception as exc:
            _raise_if_fatal_cuda_error(exc)
            raise

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
                camera_hint=plan.camera_hint,
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
                    prompt, plan, reference_image, negative_prompt, character_lock, clip_seed, local_progress
                )
                qc = self.generator.last_qc
                if qc and qc.visual_failure and plan.profile.scene_retries:
                    local_progress("Visual QC retrying opening part", 0.06)
                    retry_seed = clip_seed if clip_seed < 0 else clip_seed + 100_003
                    accepted = self._first_clip(
                        prompt, plan, reference_image, negative_prompt, character_lock, retry_seed, local_progress
                    )
            else:
                tail = _aligned_tail(previous_tail, plan.profile.tail_frames)
                overlap = max(0, len(tail) - 1)
                render_frames = plan.clip_frames + overlap
                accepted: Path

                try:
                    raw = Path(
                        self.generator.generate_conditioned_video(
                            **self._base_kwargs(
                                prompt,
                                plan,
                                negative_prompt,
                                character_lock,
                                clip_seed,
                                local_progress,
                                frames=render_frames,
                            ),
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
                    _raise_if_fatal_cuda_error(exc)
                    local_progress(f"Using compatible continuation fallback ({type(exc).__name__})", 0.08)
                    if not tail:
                        raise RuntimeError(f"Continuation failed: {exc}") from exc
                    accepted = self._i2v_fallback(
                        prompt,
                        plan,
                        tail[max(0, len(tail) - 5)],
                        negative_prompt,
                        character_lock,
                        clip_seed,
                        local_progress,
                    )

                qc = self.generator.last_qc
                if qc and qc.visual_failure and plan.profile.scene_retries:
                    local_progress("Visual QC retrying from a stable continuation frame", 0.06)
                    retry_seed = clip_seed if clip_seed < 0 else clip_seed + 100_003
                    accepted = self._i2v_fallback(
                        prompt,
                        plan,
                        tail[max(0, len(tail) - 5)],
                        negative_prompt,
                        character_lock,
                        retry_seed,
                        local_progress,
                    )

            qc = self.generator.last_qc
            if qc and (qc.fatal or qc.visual_failure):
                raise RuntimeError(
                    f"Part {index + 1} did not pass visual quality checks after retry.\n{qc.summary()}"
                )

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
