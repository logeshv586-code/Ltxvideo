"""Memory-aware LTX-Video generator for RTX 4050 laptops.

The pipeline is loaded with 8-bit transformer + T5 weights and a strict
GPU/CPU memory budget. The model is downloaded automatically by Hugging Face
on first use and then reused from the local cache for offline generation.
"""
from __future__ import annotations

import gc
import random
from datetime import datetime
from pathlib import Path
from typing import Callable

import torch
from PIL import Image

from config import (
    CPU_MEMORY_BUDGET,
    DEFAULT_FPS,
    ENABLE_8BIT,
    ENABLE_VAE_TILING,
    GPU_MEMORY_BUDGET,
    HF_REPO_ID,
    MAX_NATIVE_FRAMES,
    MODELS_DIR,
    OUTPUTS_DIR,
)

Progress = Callable[[str, float], None] | None


class VideoGenerator:
    """LTX text/image-to-video with aggressive memory control."""

    def __init__(self) -> None:
        self.pipe = None
        self.mode: str | None = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def _report(self, callback: Progress, message: str, progress: float) -> None:
        print(f"[{progress * 100:5.1f}%] {message}")
        if callback:
            callback(message, progress)

    @staticmethod
    def _validate_shape(width: int, height: int, num_frames: int) -> None:
        if width % 32 or height % 32:
            raise ValueError("LTX width and height must be divisible by 32.")
        if (num_frames - 1) % 8:
            raise ValueError("LTX frame count must follow 8k+1 (49, 97, 121, ...).")
        if num_frames > MAX_NATIVE_FRAMES:
            raise ValueError(
                f"RTX 4050 mode caps one native clip at {MAX_NATIVE_FRAMES} frames. "
                "Use Cartoon Story mode for longer videos."
            )

    def _load_pipeline(self, mode: str, progress_callback: Progress = None) -> None:
        if self.pipe is not None and self.mode == mode:
            return
        self.unload_model()

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU not detected. This profile requires an NVIDIA RTX GPU.")

        self._report(progress_callback, "Loading quantized LTX 2B pipeline…", 0.05)

        try:
            from diffusers import (
                BitsAndBytesConfig as DiffusersBitsAndBytesConfig,
                LTXImageToVideoPipeline,
                LTXPipeline,
                LTXVideoTransformer3DModel,
            )
            from transformers import BitsAndBytesConfig, T5EncoderModel
        except ImportError as exc:
            raise RuntimeError(
                "Missing LTX runtime packages. Run `python run.py` once to install dependencies."
            ) from exc

        if not ENABLE_8BIT:
            raise RuntimeError("This build expects 8-bit mode for RTX 4050 hardware.")

        max_memory = {0: GPU_MEMORY_BUDGET, "cpu": CPU_MEMORY_BUDGET}
        offload_folder = str(MODELS_DIR / "offload")

        self._report(progress_callback, "Loading T5 text encoder in 8-bit…", 0.12)
        text_encoder = T5EncoderModel.from_pretrained(
            HF_REPO_ID,
            subfolder="text_encoder",
            quantization_config=BitsAndBytesConfig(load_in_8bit=True),
            torch_dtype=torch.float16,
            device_map="auto",
            max_memory=max_memory,
            offload_folder=offload_folder,
            low_cpu_mem_usage=True,
        )

        self._report(progress_callback, "Loading 2B video transformer in 8-bit…", 0.24)
        transformer = LTXVideoTransformer3DModel.from_pretrained(
            HF_REPO_ID,
            subfolder="transformer",
            quantization_config=DiffusersBitsAndBytesConfig(load_in_8bit=True),
            torch_dtype=torch.float16,
            device_map="auto",
            max_memory=max_memory,
            offload_folder=offload_folder,
            low_cpu_mem_usage=True,
        )

        pipeline_cls = LTXImageToVideoPipeline if mode == "i2v" else LTXPipeline
        self._report(progress_callback, f"Building {mode.upper()} pipeline…", 0.38)
        self.pipe = pipeline_cls.from_pretrained(
            HF_REPO_ID,
            text_encoder=text_encoder,
            transformer=transformer,
            torch_dtype=torch.float16,
            device_map="balanced",
            max_memory=max_memory,
            offload_folder=offload_folder,
            low_cpu_mem_usage=True,
        )

        if ENABLE_VAE_TILING and hasattr(self.pipe.vae, "enable_tiling"):
            self.pipe.vae.enable_tiling()
        if hasattr(self.pipe.vae, "enable_slicing"):
            self.pipe.vae.enable_slicing()

        self.mode = mode
        self._report(progress_callback, "LTX pipeline ready.", 0.45)

    def unload_model(self) -> None:
        if self.pipe is not None:
            del self.pipe
        self.pipe = None
        self.mode = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    @staticmethod
    def _seed(seed: int) -> int:
        return random.randint(0, 2**31 - 1) if seed is None or int(seed) < 0 else int(seed)

    @staticmethod
    def _output_path(prefix: str) -> Path:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        return OUTPUTS_DIR / f"{prefix}_{stamp}.mp4"

    def generate_text_to_video(
        self,
        prompt: str,
        negative_prompt: str,
        width: int,
        height: int,
        num_frames: int,
        num_inference_steps: int,
        guidance_scale: float,
        seed: int = -1,
        progress_callback: Progress = None,
    ) -> Path:
        from diffusers.utils import export_to_video

        self._validate_shape(width, height, num_frames)
        self._load_pipeline("t2v", progress_callback)
        actual_seed = self._seed(seed)
        self._report(progress_callback, f"Generating {num_frames} frames (seed {actual_seed})…", 0.5)

        generator = torch.Generator(device="cuda").manual_seed(actual_seed)
        result = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_frames=num_frames,
            num_inference_steps=int(num_inference_steps),
            guidance_scale=float(guidance_scale),
            decode_timestep=0.05,
            decode_noise_scale=0.025,
            generator=generator,
        ).frames[0]

        output = self._output_path("ltx_t2v")
        export_to_video(result, str(output), fps=DEFAULT_FPS)
        self._report(progress_callback, f"Saved {output.name}", 1.0)
        return output

    def generate_image_to_video(
        self,
        prompt: str,
        image: Image.Image,
        negative_prompt: str,
        width: int,
        height: int,
        num_frames: int,
        num_inference_steps: int,
        guidance_scale: float,
        seed: int = -1,
        progress_callback: Progress = None,
    ) -> Path:
        from diffusers.utils import export_to_video

        self._validate_shape(width, height, num_frames)
        if image is None:
            raise ValueError("Reference image is required for image-to-video.")
        self._load_pipeline("i2v", progress_callback)
        actual_seed = self._seed(seed)
        image = image.convert("RGB").resize((width, height), Image.Resampling.LANCZOS)
        self._report(progress_callback, f"Animating reference image (seed {actual_seed})…", 0.5)

        generator = torch.Generator(device="cuda").manual_seed(actual_seed)
        result = self.pipe(
            image=image,
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_frames=num_frames,
            frame_rate=DEFAULT_FPS,
            num_inference_steps=int(num_inference_steps),
            guidance_scale=float(guidance_scale),
            decode_timestep=0.05,
            decode_noise_scale=0.025,
            generator=generator,
        ).frames[0]

        output = self._output_path("ltx_i2v")
        export_to_video(result, str(output), fps=DEFAULT_FPS)
        self._report(progress_callback, f"Saved {output.name}", 1.0)
        return output
