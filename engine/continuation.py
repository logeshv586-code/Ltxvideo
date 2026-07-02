"""
LTX-2.3 Video Generation Platform — Multi-Clip Continuation System
Generates long videos (up to 30s) by chaining 10-second clips with
visual continuity via last-frame conditioning.
"""

import sys
import time
from pathlib import Path
from typing import Callable

from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import CROSSFADE_DURATION, DEFAULT_FPS, MAX_CONTINUATION_CLIPS, OUTPUTS_DIR
from engine.generator import VideoGenerator
from engine.memory_manager import clear_gpu_memory
from engine.video_processor import concatenate_videos, extract_last_frame, get_video_info


class ContinuationGenerator:
    """
    Generates long videos by chaining multiple clips with visual continuity.
    Each subsequent clip uses the last frame of the previous clip as its
    starting image (I2V conditioning), maintaining scene consistency.
    """

    def __init__(self, generator: VideoGenerator | None = None) -> None:
        self.generator = generator or VideoGenerator()

    def generate_continuation(
        self,
        prompt: str,
        num_clips: int = 3,
        negative_prompt: str = "",
        width: int = 512,
        height: int = 320,
        frames_per_clip: int = 121,
        num_inference_steps: int = 8,
        guidance_scale: float = 3.0,
        seed: int = -1,
        first_image: Image.Image | None = None,
        progress_callback: Callable | None = None,
    ) -> Path:
        """
        Generate a long video using multi-clip continuation.

        Flow:
        1. Clip 1: Generate from text prompt (or image if provided)
        2. Clip 2..N: Extract last frame of previous clip → I2V generation
        3. Concatenate all clips with crossfade transitions

        Args:
            prompt: Base text prompt describing the video
            num_clips: Number of 10-second clips (1-3, max 30s)
            negative_prompt: What to avoid
            width: Video width (divisible by 32)
            height: Video height (divisible by 32)
            frames_per_clip: Frames per clip (8k+1 pattern)
            num_inference_steps: Denoising steps per clip
            guidance_scale: CFG scale
            seed: Starting seed (-1 for random)
            first_image: Optional image for first clip (I2V mode)
            progress_callback: Progress callback(message, overall_progress)

        Returns:
            Path to the final concatenated video
        """
        num_clips = min(num_clips, MAX_CONTINUATION_CLIPS)
        clip_paths: list[Path] = []
        total_start = time.time()

        self._report(progress_callback, "Starting multi-clip generation...", 0.0)

        # Ensure model is loaded
        if not self.generator.is_loaded:
            self._report(progress_callback, "Loading model...", 0.02)
            self.generator.load_model()

        for clip_idx in range(num_clips):
            clip_num = clip_idx + 1
            clip_progress_base = clip_idx / num_clips
            clip_progress_scale = 1.0 / num_clips

            self._report(
                progress_callback,
                f"Generating clip {clip_num}/{num_clips}...",
                clip_progress_base + 0.01,
            )

            # Build clip-specific prompt
            clip_prompt = self._build_clip_prompt(prompt, clip_idx, num_clips)

            # Create a per-clip progress wrapper
            def clip_progress(msg: str, prog: float, _base=clip_progress_base, _scale=clip_progress_scale) -> None:
                overall = _base + prog * _scale * 0.9  # Leave 10% for concat
                self._report(progress_callback, f"[Clip {clip_num}/{num_clips}] {msg}", overall)

            try:
                if clip_idx == 0:
                    # First clip: T2V or I2V if image provided
                    if first_image is not None:
                        clip_path = self.generator.generate_image_to_video(
                            prompt=clip_prompt,
                            image=first_image,
                            negative_prompt=negative_prompt,
                            width=width,
                            height=height,
                            num_frames=frames_per_clip,
                            num_inference_steps=num_inference_steps,
                            guidance_scale=guidance_scale,
                            seed=seed,
                            progress_callback=clip_progress,
                        )
                    else:
                        clip_path = self.generator.generate_text_to_video(
                            prompt=clip_prompt,
                            negative_prompt=negative_prompt,
                            width=width,
                            height=height,
                            num_frames=frames_per_clip,
                            num_inference_steps=num_inference_steps,
                            guidance_scale=guidance_scale,
                            seed=seed,
                            progress_callback=clip_progress,
                        )
                else:
                    # Subsequent clips: Extract last frame → I2V
                    self._report(
                        progress_callback,
                        f"Extracting context frame from clip {clip_num - 1}...",
                        clip_progress_base,
                    )
                    last_frame = extract_last_frame(clip_paths[-1])

                    # Use a different seed for variety but with continuity
                    clip_seed = seed + clip_idx if seed != -1 else -1

                    clip_path = self.generator.generate_image_to_video(
                        prompt=clip_prompt,
                        image=last_frame,
                        negative_prompt=negative_prompt,
                        width=width,
                        height=height,
                        num_frames=frames_per_clip,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale,
                        seed=clip_seed,
                        progress_callback=clip_progress,
                    )

                clip_paths.append(clip_path)
                info = get_video_info(clip_path)
                print(f"  ✅ Clip {clip_num} saved: {info['duration']:.1f}s, {info['width']}x{info['height']}")

                # Clear memory between clips
                clear_gpu_memory()

            except Exception as e:
                print(f"  ❌ Clip {clip_num} failed: {e}")
                self._report(
                    progress_callback,
                    f"Clip {clip_num} failed: {e}",
                    -1,
                )
                # If at least one clip was generated, try to use what we have
                if len(clip_paths) > 0:
                    break
                raise

        if len(clip_paths) == 0:
            raise RuntimeError("No clips were generated")

        # Concatenate clips
        self._report(progress_callback, "Concatenating clips with crossfade...", 0.92)

        if len(clip_paths) == 1:
            final_path = clip_paths[0]
        else:
            crossfade_frames = int(CROSSFADE_DURATION * DEFAULT_FPS)
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            final_path = OUTPUTS_DIR / f"continuation_{timestamp}.mp4"

            concatenate_videos(
                video_paths=clip_paths,
                output_path=final_path,
                crossfade_frames=crossfade_frames,
                target_fps=DEFAULT_FPS,
            )

        total_elapsed = time.time() - total_start
        final_info = get_video_info(final_path)

        self._report(
            progress_callback,
            f"Complete! {len(clip_paths)} clips → {final_info['duration']:.1f}s video ({total_elapsed:.0f}s)",
            1.0,
        )

        print(f"\n🎉 Multi-clip generation complete!")
        print(f"   Clips: {len(clip_paths)}")
        print(f"   Duration: {final_info['duration']:.1f}s")
        print(f"   Total time: {total_elapsed:.1f}s")
        print(f"   Output: {final_path}")

        return final_path

    @staticmethod
    def _build_clip_prompt(base_prompt: str, clip_index: int, total_clips: int) -> str:
        """
        Build a clip-specific prompt that maintains narrative continuity.
        The first clip uses the original prompt. Subsequent clips add
        continuation context to guide coherent generation.
        """
        if clip_index == 0:
            return base_prompt

        # For continuation clips, add context
        continuity_phrases = [
            "continuing the scene smoothly,",
            "the action continues naturally,",
            "seamlessly extending the previous moment,",
        ]

        position_phrases = {
            1: "the scene progresses further,",
            2: "reaching the climax of the scene,",
        }

        phrase = continuity_phrases[clip_index % len(continuity_phrases)]
        position = position_phrases.get(clip_index, "the scene continues,")

        return f"{phrase} {position} {base_prompt}"

    @staticmethod
    def _report(callback: Callable | None, message: str, progress: float) -> None:
        """Report progress to callback if available."""
        if callback:
            callback(message, progress)


if __name__ == "__main__":
    # Quick test
    gen = VideoGenerator()
    cont = ContinuationGenerator(gen)

    print("Testing continuation system...")
    result = cont.generate_continuation(
        prompt="A cute cartoon rabbit hopping through a sunlit meadow with flowers",
        num_clips=2,
        width=256,
        height=256,
        frames_per_clip=49,
        num_inference_steps=4,
        guidance_scale=1.0,
        seed=42,
    )
    print(f"Result: {result}")
