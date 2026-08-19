"""Cartoon story planning and continuity generation."""
from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Callable

from PIL import Image

from config import CAMERA_PRESETS, CARTOON_STYLES, DEFAULT_FPS, MAX_STORY_SCENES, OUTPUTS_DIR
from engine.generator import VideoGenerator
from engine.video_processor import concatenate_videos_streaming, extract_last_frame

Progress = Callable[[str, float], None] | None


def split_story_beats(story: str, scene_count: int) -> list[str]:
    """Convert newline/sentence story text into exactly scene_count useful beats."""
    scene_count = max(1, min(int(scene_count), MAX_STORY_SCENES))
    raw = [line.strip(" -\t") for line in story.splitlines() if line.strip()]
    if len(raw) <= 1:
        raw = [s.strip() for s in re.split(r"(?<=[.!?])\s+", story.strip()) if s.strip()]
    if not raw:
        raw = ["The characters begin their adventure."]

    beats: list[str] = []
    for index in range(scene_count):
        if index < len(raw):
            beats.append(raw[index])
        else:
            beats.append(f"Continue naturally from the previous moment: {raw[-1]}")
    return beats


def build_scene_prompt(
    beat: str,
    scene_index: int,
    scene_count: int,
    style_name: str,
    character_bible: str,
) -> str:
    style = CARTOON_STYLES.get(style_name, style_name)
    camera = CAMERA_PRESETS[scene_index % len(CAMERA_PRESETS)]
    continuity = (
        "opening scene; establish the characters exactly as described"
        if scene_index == 0
        else "direct continuation of the previous scene; preserve identical character face, clothing, colors, proportions, props and environment"
    )
    return (
        f"Cartoon story scene {scene_index + 1} of {scene_count}. {continuity}. "
        f"Story action: {beat}. Character bible: {character_bible}. "
        f"Visual style: {style}. Camera: {camera}. "
        "Clear readable action, stable anatomy, consistent design, smooth motion, no cuts inside the shot."
    )


def storyboard_markdown(story: str, scene_count: int, style_name: str, character_bible: str) -> str:
    beats = split_story_beats(story, scene_count)
    lines = [f"### 🎞️ Storyboard · {len(beats)} scenes", ""]
    for i, beat in enumerate(beats):
        lines.append(f"**Scene {i + 1}** — {beat}")
    lines += ["", f"**Style:** {style_name}", f"**Character lock:** {character_bible or 'Not supplied'}"]
    return "\n\n".join(lines)


class CartoonStoryGenerator:
    def __init__(self, generator: VideoGenerator | None = None) -> None:
        self.generator = generator or VideoGenerator()

    def generate(
        self,
        story: str,
        character_bible: str,
        style_name: str,
        scene_count: int,
        reference_image: Image.Image | None,
        width: int,
        height: int,
        frames_per_scene: int,
        num_inference_steps: int,
        guidance_scale: float,
        negative_prompt: str,
        seed: int,
        progress_callback: Progress = None,
    ) -> Path:
        beats = split_story_beats(story, scene_count)
        clips: list[Path] = []
        previous_frame = reference_image

        for idx, beat in enumerate(beats):
            prompt = build_scene_prompt(beat, idx, len(beats), style_name, character_bible)
            if progress_callback:
                progress_callback(f"Scene {idx + 1}/{len(beats)}: preparing continuity", idx / len(beats))

            scene_seed = -1 if seed < 0 else int(seed) + idx
            if previous_frame is None and idx == 0:
                clip = self.generator.generate_text_to_video(
                    prompt, negative_prompt, width, height, frames_per_scene,
                    num_inference_steps, guidance_scale, scene_seed,
                    progress_callback=None,
                )
            else:
                clip = self.generator.generate_image_to_video(
                    prompt, previous_frame, negative_prompt, width, height, frames_per_scene,
                    num_inference_steps, guidance_scale, scene_seed,
                    progress_callback=None,
                )

            clips.append(Path(clip))
            previous_frame = extract_last_frame(clip)
            if progress_callback:
                progress_callback(f"Scene {idx + 1}/{len(beats)} complete", (idx + 0.9) / len(beats))

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output = OUTPUTS_DIR / f"cartoon_story_{timestamp}.mp4"
        concatenate_videos_streaming(clips, output, target_fps=DEFAULT_FPS)
        if progress_callback:
            progress_callback("Cartoon story assembled", 1.0)
        return output
