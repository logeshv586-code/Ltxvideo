"""Wan2.1-T2V-1.3B backend for short cartoon-friendly video clips.

Wan is deliberately kept at its official stable 832x480 generation size.  On
low-VRAM hardware such as an RTX 4050 6 GB, quality is improved after diffusion
with optional RIFE motion interpolation and Real-ESRGAN cartoon upscaling.
"""
from __future__ import annotations

import os
import random
from datetime import datetime
from pathlib import Path
from typing import Callable

import torch

from config import MODELS_DIR, OUTPUTS_DIR
from engine.cogvideox_generator import build_story_sequence
from engine.video_enhancement import enhance_wan_delivery
from engine.video_processor import concatenate_videos_streaming, export_delivery

Progress = Callable[[str, float], None] | None
WAN_REPO_ID = "Wan-AI/Wan2.1-T2V-1.3B"
WAN_MODEL_DIR = Path(os.getenv("WAN_MODEL_DIR", MODELS_DIR / "wan2.1-t2v-1.3b"))
WAN_WIDTH = 832
WAN_HEIGHT = 480
WAN_FRAMES = 81
WAN_FPS = 16  # Native model FPS; kept for backwards compatibility/tests.
WAN_DELIVERY_FPS = int(os.getenv("WAN_DELIVERY_FPS", "32"))
WAN_STEPS = int(os.getenv("WAN_STEPS", "40"))
WAN_CLIP_SECONDS = WAN_FRAMES / WAN_FPS


def select_wan_offload_mode(total_vram_gb: float, requested: str | None = None) -> str:
    """Use sequential offload below 12 GB so a 6 GB laptop can still run."""
    choice = (requested or os.getenv("WAN_OFFLOAD_MODE", "auto")).strip().lower()
    if choice in {"model", "sequential"}:
        return choice
    return "sequential" if total_vram_gb < 12 else "model"


class WanVideoGenerator:
    """Lazy-loaded Wan Diffusers pipeline, one GPU at a time."""

    def __init__(self, device_index: int = 0) -> None:
        self.device_index = int(device_index)
        self.pipe = None

    @staticmethod
    def _report(callback: Progress, message: str, value: float) -> None:
        if callback:
            callback(message, min(1.0, max(0.0, float(value))))

    @staticmethod
    def _seed(seed: int | float) -> int:
        return random.randint(0, 2**31 - 1) if int(seed) < 0 else int(seed)

    def _clear_cuda_cache(self) -> None:
        if torch.cuda.is_available():
            with torch.cuda.device(self.device_index):
                torch.cuda.empty_cache()
                if hasattr(torch.cuda, "ipc_collect"):
                    torch.cuda.ipc_collect()

    def readiness(self) -> tuple[bool, str]:
        required = ("model_index.json", "transformer", "text_encoder", "vae")
        missing = [item for item in required if not (WAN_MODEL_DIR / item).exists()]
        if missing:
            return False, f"Setup required at {WAN_MODEL_DIR}; missing: {', '.join(missing)}"
        return True, (
            f"Wan2.1-T2V-1.3B ready: native {WAN_WIDTH}x{WAN_HEIGHT}, "
            f"{WAN_FRAMES} frames at {WAN_FPS} FPS; enhanced delivery targets {WAN_DELIVERY_FPS} FPS."
        )

    def _load(self, callback: Progress) -> None:
        if self.pipe is not None:
            return
        if not torch.cuda.is_available():
            raise RuntimeError("Wan2.1 requires an NVIDIA CUDA GPU.")
        ready, message = self.readiness()
        if not ready:
            raise RuntimeError(f"{message}. Run: python setup_wan.py --download")
        try:
            from diffusers import WanPipeline
        except ImportError as exc:
            raise RuntimeError("Wan2.1 needs a current Diffusers installation with WanPipeline support.") from exc

        total_bytes = torch.cuda.get_device_properties(self.device_index).total_memory
        mode = select_wan_offload_mode(total_bytes / (1024**3))
        self._report(callback, f"GPU {self.device_index}: loading Wan2.1-T2V-1.3B", 0.05)
        self.pipe = WanPipeline.from_pretrained(
            str(WAN_MODEL_DIR),
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )
        if mode == "sequential":
            self.pipe.enable_sequential_cpu_offload(gpu_id=self.device_index)
            offload_label = "6 GB-safe sequential CPU offload"
        else:
            self.pipe.enable_model_cpu_offload(gpu_id=self.device_index)
            offload_label = "model CPU offload"
        if hasattr(self.pipe.vae, "enable_tiling"):
            self.pipe.vae.enable_tiling()
        if hasattr(self.pipe.vae, "enable_slicing"):
            self.pipe.vae.enable_slicing()
        self._clear_cuda_cache()
        self._report(callback, f"Wan2.1 pipeline ready ({offload_label})", 0.16)

    @staticmethod
    def _quality_prompt(prompt: str) -> str:
        """Keep the cartoon render stable without overloading the text prompt."""
        return (
            f"{prompt.strip()}. Premium family animation for children, stable character identity, "
            "same facial features and body proportions throughout the shot, consistent costume and colors, "
            "clean silhouette, controlled readable motion, coherent background geometry, polished soft lighting, "
            "clean anatomy, no text, watermark, flicker, jitter, morphing or character redesign."
        )

    def generate(self, prompt: str, seed: int | float = -1, callback: Progress = None) -> Path:
        if not prompt or not prompt.strip():
            raise ValueError("Describe the Wan2.1 clip first.")
        self._load(callback)
        actual_seed = self._seed(seed)
        enhanced_prompt = self._quality_prompt(prompt)
        self._report(callback, f"Wan2.1 rendering {WAN_FRAMES} native frames on GPU {self.device_index}", 0.20)
        self._clear_cuda_cache()
        try:
            with torch.cuda.device(self.device_index), torch.inference_mode():
                frames = self.pipe(
                    prompt=enhanced_prompt,
                    negative_prompt=(
                        "blurry, low detail, distorted, flicker, jitter, morphing, identity drift, character redesign, "
                        "costume change, color shift, duplicate character, extra limbs, broken anatomy, text, watermark"
                    ),
                    height=WAN_HEIGHT,
                    width=WAN_WIDTH,
                    num_frames=WAN_FRAMES,
                    num_inference_steps=WAN_STEPS,
                    guidance_scale=6.0,
                    generator=torch.Generator(device=f"cuda:{self.device_index}").manual_seed(actual_seed),
                ).frames[0]
        except torch.OutOfMemoryError as exc:
            self._clear_cuda_cache()
            raise RuntimeError(
                "Wan2.1 exhausted this GPU. Use WAN_OFFLOAD_MODE=sequential and close other GPU jobs before retrying."
            ) from exc

        from diffusers.utils import export_to_video

        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        native = OUTPUTS_DIR / f"wan_native_{stamp}.mp4"
        final = OUTPUTS_DIR / f"wan_youtube_1080p_{WAN_DELIVERY_FPS}fps_{stamp}.mp4"
        self._report(callback, "Encoding native Wan2.1 clip", 0.90)
        export_to_video(frames, str(native), fps=WAN_FPS)
        del frames
        self._clear_cuda_cache()

        def enhancement_status(message: str) -> None:
            self._report(callback, message, 0.96)

        self._report(callback, "Preparing smooth enhanced 1080p delivery", 0.93)
        enhance_wan_delivery(
            native,
            final,
            width=1920,
            height=1080,
            native_fps=WAN_FPS,
            target_fps=WAN_DELIVERY_FPS,
            gpu_id=self.device_index,
            callback=enhancement_status,
        )
        self._report(callback, f"Saved {final.name}", 1.0)
        return final

    def generate_story(
        self,
        story: str,
        target_seconds: int,
        seed: int | float = -1,
        callback: Progress = None,
    ) -> tuple[Path, tuple[str, ...]]:
        beats = build_story_sequence(story, target_seconds)
        clips: list[Path] = []
        context = " ".join(story.split())[:700]
        total = len(beats)
        # One base seed is intentionally reused for every story scene.  The
        # event text changes, but the noise anchor stays stable to reduce
        # character/wardrobe redesign between independently generated clips.
        base_seed = self._seed(seed)
        continuity_lock = (
            "Continuity lock: the exact same main character identity, face, age, body proportions, hairstyle/fur, "
            "costume, colors and art style must persist in every scene. Preserve the same world design and lighting "
            "language. Do not redesign the character between shots."
        )

        for index, beat in enumerate(beats):
            scene_prompt = (
                f"Story clip {index + 1} of {total}. Full story context: {context}. "
                f"Current event only: {beat}. {continuity_lock} "
                "Keep motion simple and readable: one main action and one controlled camera move."
            )

            def scene_callback(message: str, value: float, scene_index: int = index) -> None:
                if callback:
                    callback(f"Scene {scene_index + 1}/{total}: {message}", (scene_index + value) / total)

            clips.append(self.generate(scene_prompt, base_seed, scene_callback))

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        assembled = OUTPUTS_DIR / f"wan_story_assembled_{stamp}.mp4"
        final = OUTPUTS_DIR / f"wan_story_1080p_{WAN_DELIVERY_FPS}fps_{stamp}.mp4"
        self._report(callback, "Combining ordered enhanced Wan2.1 clips", 0.995)
        concatenate_videos_streaming(clips, assembled, target_fps=WAN_DELIVERY_FPS)
        export_delivery(
            assembled,
            final,
            1920,
            1080,
            enhance_quality=True,
            target_fps=WAN_DELIVERY_FPS,
            duration_seconds=target_seconds,
        )
        self._report(callback, f"Saved {final.name}", 1.0)
        return final, beats


GENERATOR = WanVideoGenerator(device_index=int(os.getenv("WAN_GPU_ID", "0")))
