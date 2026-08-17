from __future__ import annotations

import gc
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Callable

import psutil
import torch
from PIL import Image

from config import (
    DEFAULT_FPS,
    ENABLE_FP8,
    GEMMA_DIR,
    IMAGE_CONDITIONING_STRENGTH,
    KEEP_MODEL_LOADED,
    MODELS_DIR,
    OFFLOAD_MODE,
    OUTPUTS_DIR,
)
from engine.memory_manager import clear_gpu_memory, setup_memory_optimizations

try:
    from ltx_core.model.video_vae import AUTO_TILING, get_video_chunks_number
    from ltx_pipelines.distilled import DistilledPipeline
    from ltx_pipelines.utils.args import ImageConditioningInput
    from ltx_pipelines.utils.media_io import encode_video
    from ltx_pipelines.utils.model_paths import ModelPaths
    from ltx_pipelines.utils.quantization_factory import QuantizationKind
    from ltx_pipelines.utils.types import OffloadMode
    LTX_PIPELINES_AVAILABLE = True
    LTX_IMPORT_ERROR = None
except Exception as exc:  # surfaced as a friendly setup error later
    LTX_PIPELINES_AVAILABLE = False
    LTX_IMPORT_ERROR = exc

ProgressCallback = Callable[[str, float], None]


class VideoGenerator:
    """Fast LTX-2.3 distilled T2V/I2V engine tuned for a 16 GB GPU."""

    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pipe = None
        self.is_loaded = False
        self.offload_mode = None
        setup_memory_optimizations()

    def _report(self, message: str, progress: float, callback: ProgressCallback | None = None) -> None:
        print(f"[{progress * 100:5.1f}%] {message}")
        if callback:
            callback(message, progress)

    def _required_paths(self) -> tuple[Path, Path, Path]:
        checkpoint = MODELS_DIR / "ltx-2.3-22b-distilled-1.1.safetensors"
        upscaler = MODELS_DIR / "ltx-2.3-spatial-upscaler-x2-1.1.safetensors"
        gemma = GEMMA_DIR
        missing = [str(p) for p in (checkpoint, upscaler, gemma) if not p.exists()]
        if missing:
            raise FileNotFoundError(
                "Missing LTX model assets:\n- " + "\n- ".join(missing) +
                "\nRun: python download_models.py"
            )
        return checkpoint, upscaler, gemma

    def _choose_offload_mode(self):
        if OFFLOAD_MODE == "cpu":
            return OffloadMode.CPU
        if OFFLOAD_MODE == "disk":
            return OffloadMode.DISK
        if OFFLOAD_MODE == "none":
            return OffloadMode.NONE

        # Official LTX docs describe CPU offload as requiring roughly 36 GB RAM.
        # Prefer it when available because repeated generations reuse the CPU cache.
        ram_gb = psutil.virtual_memory().total / (1024 ** 3)
        if ram_gb >= 40:
            return OffloadMode.CPU
        return OffloadMode.DISK

    def load_model(self, progress_callback: ProgressCallback | None = None) -> None:
        if self.is_loaded:
            return
        if self.device.type != "cuda":
            raise RuntimeError("A CUDA NVIDIA GPU is required for this local LTX setup.")
        if not LTX_PIPELINES_AVAILABLE:
            raise RuntimeError(
                "Official LTX packages are not installed or incompatible. "
                "Run: python setup_ltx.py\n"
                f"Import error: {LTX_IMPORT_ERROR}"
            )

        checkpoint, upscaler, gemma = self._required_paths()
        self.offload_mode = self._choose_offload_mode()
        self._report(
            f"Loading LTX-2.3 distilled pipeline ({self.offload_mode.value} offload)...",
            0.08,
            progress_callback,
        )

        model_paths = ModelPaths.from_monolith(
            checkpoint_path=str(checkpoint),
            gemma_root=str(gemma),
        )
        quantization = (
            QuantizationKind("fp8-cast").to_policy(checkpoint_path=str(checkpoint))
            if ENABLE_FP8 else None
        )

        self.pipe = DistilledPipeline(
            model_paths=model_paths,
            spatial_upsampler_path=str(upscaler),
            loras=[],
            device=self.device,
            quantization=quantization,
            offload_mode=self.offload_mode,
        )
        self.is_loaded = True
        self._report("LTX model loaded and cached for repeated generations.", 0.18, progress_callback)

    def unload_model(self) -> None:
        if self.pipe is not None:
            del self.pipe
        self.pipe = None
        self.is_loaded = False
        gc.collect()
        clear_gpu_memory()

    def _output_path(self, prefix: str) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        return OUTPUTS_DIR / f"{prefix}_{timestamp}.mp4"

    def _run(
        self,
        *,
        prompt: str,
        width: int,
        height: int,
        num_frames: int,
        seed: int,
        images: list,
        prefix: str,
        progress_callback: ProgressCallback | None = None,
    ) -> Path:
        if seed < 0:
            seed = int(torch.randint(0, 2**31 - 1, (1,)).item())

        self.load_model(progress_callback)
        clear_gpu_memory()
        self._report(f"Generating {width}×{height}, {num_frames} frames, seed {seed}...", 0.25, progress_callback)

        try:
            video, audio, resolved_frames, tiling_config = self.pipe(
                prompt=prompt,
                seed=seed,
                height=height,
                width=width,
                num_frames=num_frames,
                frame_rate=float(DEFAULT_FPS),
                images=images,
                tiling_config=AUTO_TILING,
                enhance_prompt=False,
            )

            output_path = self._output_path(prefix)
            self._report("Encoding MP4 with synchronized LTX audio...", 0.90, progress_callback)
            encode_video(
                video=video,
                fps=DEFAULT_FPS,
                audio=audio,
                output_path=str(output_path),
                video_chunks_number=get_video_chunks_number(resolved_frames, tiling_config),
            )
            self._report(f"Saved {output_path.name}", 1.0, progress_callback)
            return output_path
        finally:
            clear_gpu_memory()
            if not KEEP_MODEL_LOADED:
                self.unload_model()

    def generate_text_to_video(
        self,
        prompt: str,
        negative_prompt: str = "",
        width: int = 512,
        height: int = 320,
        num_frames: int = 97,
        num_inference_steps: int = 8,
        guidance_scale: float = 3.0,
        seed: int = -1,
        progress_callback: ProgressCallback | None = None,
    ) -> Path:
        # LTX DistilledPipeline uses its trained fixed sigma schedule. The legacy UI
        # knobs are accepted for compatibility but intentionally do not override it.
        _ = negative_prompt, num_inference_steps, guidance_scale
        if not prompt.strip():
            raise ValueError("Prompt cannot be empty.")
        return self._run(
            prompt=prompt.strip(),
            width=int(width),
            height=int(height),
            num_frames=int(num_frames),
            seed=int(seed),
            images=[],
            prefix="t2v",
            progress_callback=progress_callback,
        )

    def generate_image_to_video(
        self,
        prompt: str,
        image: Image.Image,
        negative_prompt: str = "",
        width: int = 512,
        height: int = 320,
        num_frames: int = 97,
        num_inference_steps: int = 8,
        guidance_scale: float = 3.0,
        seed: int = -1,
        progress_callback: ProgressCallback | None = None,
    ) -> Path:
        _ = negative_prompt, num_inference_steps, guidance_scale
        if image is None:
            raise ValueError("Image is required for image-to-video generation.")

        temp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                temp_path = Path(tmp.name)
            image.convert("RGB").save(temp_path)
            conditioning = ImageConditioningInput(
                path=str(temp_path),
                frame_idx=0,
                strength=IMAGE_CONDITIONING_STRENGTH,
                crf=None,
            )
            return self._run(
                prompt=prompt.strip(),
                width=int(width),
                height=int(height),
                num_frames=int(num_frames),
                seed=int(seed),
                images=[conditioning],
                prefix="i2v",
                progress_callback=progress_callback,
            )
        finally:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)

    def test(self) -> None:
        result = self.generate_text_to_video(
            prompt="A blue sphere rotates slowly on a clean white background, static camera.",
            width=384,
            height=256,
            num_frames=49,
            seed=42,
        )
        print(f"✅ Integration test output: {result}")


if __name__ == "__main__":
    VideoGenerator().test()
