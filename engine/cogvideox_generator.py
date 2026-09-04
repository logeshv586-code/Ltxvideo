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
import re
import threading
from math import ceil
from datetime import datetime
from pathlib import Path
from typing import Callable

import torch

from config import MODELS_DIR, OUTPUTS_DIR
from engine.video_processor import concatenate_videos_streaming, export_delivery

Progress = Callable[[str, float], None] | None
COGVIDEOX_REPO_ID = "zai-org/CogVideoX-5b"
COGVIDEOX_MODEL_DIR = Path(os.getenv("COGVIDEOX_MODEL_DIR", MODELS_DIR / "cogvideox-5b"))
COGVIDEOX_FPS = 8
COGVIDEOX_FRAMES = 49  # Official 6-second / 8 FPS checkpoint profile.
COGVIDEOX_STEPS = 50
COGVIDEOX_CLIP_SECONDS = COGVIDEOX_FRAMES / COGVIDEOX_FPS
# CogVideoX model offload itself needs roughly 19 GB. A 14.6 GB T4 must use
# sequential offload; it is much slower, but keeps the model and denoising
# activations from competing for the final few MB of VRAM.
COGVIDEOX_OFFLOAD_MODE = os.getenv("COGVIDEOX_OFFLOAD_MODE", "sequential").strip().lower()
_SEQUENCE_SPLIT = re.compile(
    r"\s*(?:\.|!|\?|;|\n|,\s*then\s+|\band then\b|\bafter that\b|\bafterwards\b|\bfinally\b|\bnext\b|\bthen\b)\s*",
    flags=re.IGNORECASE,
)


def build_story_sequence(story: str, target_seconds: int) -> tuple[str, ...]:
    """Split a chronological prompt into safe, ordered CogVideoX clip beats."""
    clean = re.sub(r"\s+", " ", (story or "").strip())
    if not clean:
        raise ValueError("Describe the story first.")
    clip_count = max(1, int(ceil(max(1, int(target_seconds)) / COGVIDEOX_CLIP_SECONDS)))
    beats = [part.strip(" ,.-") for part in _SEQUENCE_SPLIT.split(clean) if part.strip(" ,.-")]
    if not beats:
        beats = [clean]
    if len(beats) >= clip_count:
        grouped: list[str] = []
        for index in range(clip_count):
            start = round(index * len(beats) / clip_count)
            end = round((index + 1) * len(beats) / clip_count)
            grouped.append("; then ".join(beats[start:end]))
        return tuple(grouped)

    ordered = list(beats)
    while len(ordered) < clip_count:
        phase = len(ordered) + 1
        ordered.append(f"Continue naturally from the previous story event, progression phase {phase}: {beats[-1]}")
    return tuple(ordered[:clip_count])


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

    def _clear_cuda_cache(self) -> None:
        """Release cached tensors before a constrained-GPU render or retry."""
        if torch.cuda.is_available():
            with torch.cuda.device(self.device_index):
                torch.cuda.empty_cache()
                if hasattr(torch.cuda, "ipc_collect"):
                    torch.cuda.ipc_collect()

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

        # T4 has FP16/INT8 support but not native BF16 Tensor Cores. Sequential
        # offload is deliberate: model offload can still peak above a T4's VRAM
        # during denoising, while sequential offload leaves activation headroom.
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
        if COGVIDEOX_OFFLOAD_MODE == "model":
            self.pipe.enable_model_cpu_offload(gpu_id=self.device_index)
            offload_label = "model CPU offload"
        else:
            self.pipe.enable_sequential_cpu_offload(gpu_id=self.device_index)
            offload_label = "T4-safe sequential CPU offload"
        if hasattr(self.pipe.vae, "enable_tiling"):
            self.pipe.vae.enable_tiling()
        if hasattr(self.pipe.vae, "enable_slicing"):
            self.pipe.vae.enable_slicing()
        self._clear_cuda_cache()
        self._report(callback, f"CogVideoX pipeline ready ({offload_label})", 0.16)

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
        self._clear_cuda_cache()
        try:
            with torch.cuda.device(self.device_index), torch.inference_mode():
                frames = self.pipe(
                    prompt=enhanced_prompt,
                    num_videos_per_prompt=1,
                    num_inference_steps=COGVIDEOX_STEPS,
                    num_frames=COGVIDEOX_FRAMES,
                    guidance_scale=6.0,
                    generator=torch.Generator(device=f"cuda:{self.device_index}").manual_seed(actual_seed),
                ).frames[0]
        except torch.OutOfMemoryError as exc:
            self._clear_cuda_cache()
            raise RuntimeError(
                "CogVideoX exhausted this GPU. Restart the server after applying the T4 sequential-offload update; "
                "do not use COGVIDEOX_OFFLOAD_MODE=model on a 16 GB card."
            ) from exc

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

    def generate_story(
        self,
        story: str,
        target_seconds: int,
        seed: int | float = -1,
        callback: Progress = None,
    ) -> tuple[Path, tuple[str, ...]]:
        """Render chronological clips sequentially, then join them into one MP4."""
        beats = build_story_sequence(story, target_seconds)
        clips: list[Path] = []
        total = len(beats)
        context = " ".join(story.split())[:600]
        for index, beat in enumerate(beats):
            scene_prompt = (
                f"Story clip {index + 1} of {total}. Full story continuity context: {context}. "
                f"Current ordered event: {beat}. Continue after the prior event; preserve the same important subjects, "
                "wardrobe, environment, lighting direction and camera language."
            )
            scene_seed = -1 if int(seed) < 0 else int(seed) + index * 17

            def scene_callback(message: str, value: float, scene_index: int = index) -> None:
                if callback:
                    callback(
                        f"Scene {scene_index + 1}/{total}: {message}",
                        (scene_index + min(1.0, max(0.0, float(value)))) / total,
                    )

            clips.append(self.generate(scene_prompt, scene_seed, scene_callback))

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        assembled = OUTPUTS_DIR / f"cogvideox_story_assembled_{stamp}.mp4"
        combined = OUTPUTS_DIR / f"cogvideox_story_1080p_{stamp}.mp4"
        if callback:
            callback("Combining ordered CogVideoX clips", 0.995)
        concatenate_videos_streaming(clips, assembled, target_fps=COGVIDEOX_FPS)
        # Individual CogVideoX shots are fixed-length. Trim after every ordered
        # scene has been joined so the delivery matches the selected duration.
        export_delivery(
            assembled,
            combined,
            1920,
            1080,
            enhance_quality=True,
            target_fps=COGVIDEOX_FPS,
            duration_seconds=target_seconds,
        )
        if callback:
            callback(f"Saved {combined.name}", 1.0)
        return combined, beats


GENERATOR = CogVideoXGeneratorPool()
