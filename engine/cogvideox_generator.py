"""Quantized CogVideoX-5B text-to-video backend for the dual-T4 server.

CogVideoX-5B is deliberately used for independent, higher-fidelity clips. Its
official 5B checkpoint is text-to-video only, so visual continuation remains
the responsibility of the LTX Fox workflow, which conditions each next shot on
tail frames from the previous one.
"""
from __future__ import annotations

import os
import queue
import random
import threading
from datetime import datetime
from pathlib import Path
from typing import Callable

import torch

from config import MODELS_DIR, OUTPUTS_DIR
from engine.video_processor import export_delivery

Progress = Callable[[str, float], None] | None
COGVIDEOX_REPO_ID = "zai-org/CogVideoX-5b"
COGVIDEOX_MODEL_DIR = Path(os.getenv("COGVIDEOX_MODEL_DIR", MODELS_DIR / "cogvideox-5b"))
COGVIDEOX_FPS = 8
COGVIDEOX_FRAMES = 49  # Official 6-second / 8 FPS checkpoint profile.
COGVIDEOX_STEPS = 50


class CogVideoXWorker:
    """One CPU-offloaded, INT8-weight CogVideoX pipeline pinned to one GPU."""

    def __init__(self, device_index: int) -> None:
        self.device_index = int(device_index)
        self.pipe = None

    @staticmethod
    def _report(callback: Progress, message: str, value: float) -> None:
        if callback:
            callback(message, min(1.0, max(0.0, float(value))))

    @staticmethod
    def _seed(seed: int | float) -> int:
        return random.randint(0, 2**31 - 1) if int(seed) < 0 else int(seed)

    def _load(self, callback: Progress) -> None:
        if self.pipe is not None:
            return
        if not torch.cuda.is_available():
            raise RuntimeError("CogVideoX requires an NVIDIA CUDA GPU.")
        if not COGVIDEOX_MODEL_DIR.exists():
            raise RuntimeError(
                f"CogVideoX-5B is not installed at {COGVIDEOX_MODEL_DIR}. "
                "Run: python setup_cogvideox.py --download"
            )

        self._report(callback, f"GPU {self.device_index}: loading CogVideoX-5B INT8 weights", 0.05)
        try:
            from diffusers import BitsAndBytesConfig as DiffusersBitsAndBytesConfig
            from diffusers import CogVideoXPipeline, CogVideoXTransformer3DModel
            from transformers import BitsAndBytesConfig, T5EncoderModel
        except ImportError as exc:
            raise RuntimeError("CogVideoX needs current Diffusers, Transformers and bitsandbytes.") from exc

        # T4 has fast FP16/INT8 support but not native BF16 Tensor Cores. Weight
        # quantization plus CPU offload makes one worker fit independently on
        # each card; it is intentionally slower than an A100/H100 reference run.
        dtype = torch.float16
        model_dir = str(COGVIDEOX_MODEL_DIR)
        text_encoder = T5EncoderModel.from_pretrained(
            model_dir,
            subfolder="text_encoder",
            torch_dtype=dtype,
            quantization_config=BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_enable_fp32_cpu_offload=True,
            ),
            low_cpu_mem_usage=True,
        )
        transformer = CogVideoXTransformer3DModel.from_pretrained(
            model_dir,
            subfolder="transformer",
            torch_dtype=dtype,
            quantization_config=DiffusersBitsAndBytesConfig(load_in_8bit=True),
            low_cpu_mem_usage=True,
        )
        self.pipe = CogVideoXPipeline.from_pretrained(
            model_dir,
            text_encoder=text_encoder,
            transformer=transformer,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )
        self.pipe.enable_model_cpu_offload(gpu_id=self.device_index)
        if hasattr(self.pipe.vae, "enable_tiling"):
            self.pipe.vae.enable_tiling()
        if hasattr(self.pipe.vae, "enable_slicing"):
            self.pipe.vae.enable_slicing()
        self._report(callback, "CogVideoX pipeline ready", 0.16)

    def generate(self, prompt: str, seed: int | float = -1, callback: Progress = None) -> Path:
        self._load(callback)
        if not prompt or not prompt.strip():
            raise ValueError("Describe the CogVideoX clip first.")
        actual_seed = self._seed(seed)
        enhanced_prompt = (
            f"{prompt.strip()}. One coherent cinematic shot, clear subject silhouette, natural motion, "
            "stable anatomy, detailed materials, controlled lighting, no text, watermark, flicker or morphing."
        )
        self._report(callback, f"CogVideoX rendering 49 native frames on GPU {self.device_index}", 0.20)
        with torch.cuda.device(self.device_index), torch.inference_mode():
            frames = self.pipe(
                prompt=enhanced_prompt,
                num_videos_per_prompt=1,
                num_inference_steps=COGVIDEOX_STEPS,
                num_frames=COGVIDEOX_FRAMES,
                guidance_scale=6.0,
                generator=torch.Generator(device=f"cuda:{self.device_index}").manual_seed(actual_seed),
            ).frames[0]

        from diffusers.utils import export_to_video

        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        native = OUTPUTS_DIR / f"cogvideox_native_{stamp}.mp4"
        final = OUTPUTS_DIR / f"cogvideox_youtube_1080p_{stamp}.mp4"
        self._report(callback, "Encoding native CogVideoX clip", 0.90)
        export_to_video(frames, str(native), fps=COGVIDEOX_FPS)
        self._report(callback, "Preparing 1080p YouTube delivery", 0.96)
        export_delivery(native, final, 1920, 1080, enhance_quality=True, target_fps=COGVIDEOX_FPS)
        self._report(callback, f"Saved {final.name}", 1.0)
        return final


class CogVideoXGeneratorPool:
    """Dispatch one isolated CogVideoX request to each physical GPU."""

    def __init__(self) -> None:
        visible = torch.cuda.device_count() if torch.cuda.is_available() else 0
        requested = int(os.getenv("COGVIDEOX_MAX_GPU_WORKERS", str(max(1, visible))))
        self.worker_count = max(1, min(max(1, visible), max(1, requested)))
        self._workers = [CogVideoXWorker(index) for index in range(self.worker_count)]
        self._available: queue.Queue[int] = queue.Queue()
        for index in range(self.worker_count):
            self._available.put(index)

    def readiness(self) -> tuple[bool, str]:
        required = ("model_index.json", "transformer", "text_encoder", "vae")
        missing = [item for item in required if not (COGVIDEOX_MODEL_DIR / item).exists()]
        if missing:
            return False, f"Setup required at {COGVIDEOX_MODEL_DIR}; missing: {', '.join(missing)}"
        return True, f"CogVideoX-5B ready: {self.worker_count} independent GPU worker(s)."

    def generate(self, prompt: str, seed: int | float = -1, callback: Progress = None) -> Path:
        index = self._available.get()
        try:
            return self._workers[index].generate(prompt, seed, callback)
        finally:
            self._available.put(index)


GENERATOR = CogVideoXGeneratorPool()
