"""Adaptive multi-GPU execution wrapper for the LTX generator.

Each physical GPU owns its own persistent ``VideoGenerator`` instance. This is
important on servers with multiple cards: two 16 GB T4s are treated as two
independent 16 GB workers, never as a fake 32 GB device. Gradio can therefore
run two generation requests concurrently while every individual render stays
inside one card's VRAM limit.

The wrapper also raises native render resolution on higher-VRAM GPUs and keeps
longer clips more conservative. The customer-facing UI does not need separate
T4/RTX presets; hardware detection happens automatically.
"""
from __future__ import annotations

import math
import os
import queue
import threading
from contextlib import contextmanager

import torch

from engine.generator import Progress, VideoGenerator
from engine.hardware_profiles import HardwareProfile, select_hardware_profile

try:
    import psutil
except Exception:  # pragma: no cover - runtime dependency is normally present
    psutil = None


class _OptimizedWorker(VideoGenerator):
    """One persistent LTX pipeline bound to one CUDA device."""

    def __init__(self, device_index: int = 0) -> None:
        self.device_index = int(device_index)
        self.vram_total_gb = 0.0
        self.ram_total_gb = 0.0
        self.gpu_name = "CUDA GPU"
        self.profile: HardwareProfile | None = None

        if psutil is not None:
            try:
                self.ram_total_gb = psutil.virtual_memory().total / 1024**3
            except Exception:
                pass

        if torch.cuda.is_available() and self.device_index < torch.cuda.device_count():
            props = torch.cuda.get_device_properties(self.device_index)
            self.gpu_name = props.name
            self.vram_total_gb = props.total_memory / 1024**3
            self.profile = select_hardware_profile(
                self.gpu_name,
                self.vram_total_gb,
                self.ram_total_gb,
            )

        super().__init__()
        if torch.cuda.is_available():
            try:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                torch.backends.cudnn.benchmark = True
                torch.set_float32_matmul_precision("high")
            except Exception:
                pass

    @contextmanager
    def _device_context(self):
        if torch.cuda.is_available():
            with torch.cuda.device(self.device_index):
                yield
        else:
            yield

    def _max_native_pixels(self, num_frames: int) -> int:
        """Pixel cap protects the VAE/attention peak, especially for 8s clips."""
        vram = self.vram_total_gb
        long_clip = int(num_frames) > 121
        if vram >= 20:
            return 390_000 if long_clip else 520_000
        if vram >= 14:
            return 280_000 if long_clip else 360_000
        if vram >= 10:
            return 230_000 if long_clip else 300_000
        if vram >= 7:
            return 190_000 if long_clip else 230_000
        return 170_000

    @staticmethod
    def _round32(value: float) -> int:
        return max(32, int(round(float(value) / 32.0)) * 32)

    def _adaptive_dimensions(self, width: int, height: int, num_frames: int) -> tuple[int, int]:
        profile = self.profile
        if profile is None:
            return int(width), int(height)

        scale = profile.long_clip_scale if int(num_frames) > 121 else profile.native_scale
        if scale <= 1.001:
            return int(width), int(height)

        new_w = self._round32(int(width) * scale)
        new_h = self._round32(int(height) * scale)
        max_pixels = self._max_native_pixels(num_frames)

        if new_w * new_h > max_pixels:
            shrink = math.sqrt(max_pixels / float(new_w * new_h))
            new_w = max(32, int((new_w * shrink) // 32) * 32)
            new_h = max(32, int((new_h * shrink) // 32) * 32)

        # Never reduce below the UI/planner request.
        new_w = max(int(width), new_w)
        new_h = max(int(height), new_h)
        # The planner already supplies dimensions divisible by 32.
        return new_w, new_h

    def _adapt_kwargs(self, kwargs: dict) -> dict:
        if "width" not in kwargs or "height" not in kwargs or "num_frames" not in kwargs:
            return kwargs
        width = int(kwargs["width"])
        height = int(kwargs["height"])
        frames = int(kwargs["num_frames"])
        new_w, new_h = self._adaptive_dimensions(width, height, frames)
        if (new_w, new_h) != (width, height):
            kwargs = dict(kwargs)
            kwargs["width"] = new_w
            kwargs["height"] = new_h
            callback = kwargs.get("progress_callback")
            if callback:
                callback(
                    f"Adaptive native quality: {width}x{height} → {new_w}x{new_h} on GPU {self.device_index} ({self.gpu_name})",
                    0.015,
                )
        return kwargs

    def generate_text_to_video(self, *args, **kwargs):
        kwargs = self._adapt_kwargs(kwargs)
        with self._device_context():
            return super().generate_text_to_video(*args, **kwargs)

    def generate_image_to_video(self, *args, **kwargs):
        kwargs = self._adapt_kwargs(kwargs)
        with self._device_context():
            return super().generate_image_to_video(*args, **kwargs)

    def generate_conditioned_video(self, *args, **kwargs):
        kwargs = self._adapt_kwargs(kwargs)
        with self._device_context():
            return super().generate_conditioned_video(*args, **kwargs)

    def unload_model(self) -> None:
        with self._device_context():
            super().unload_model()

    def _run_pipe_with_cuda_retry(self, mode: str, kwargs: dict, progress_callback: Progress):
        """Keep caches warm normally; clear only the worker's GPU after OOM."""
        self._load_pipeline(mode, progress_callback)
        run_kwargs = self._with_step_progress(kwargs, progress_callback)
        try:
            with torch.inference_mode():
                result = self.pipe(**run_kwargs)
            self._report(progress_callback, "Denoising complete; decoding video frames…", 0.91)
            return result
        except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
            message = str(exc).lower()
            is_cuda_memory_error = (
                isinstance(exc, torch.cuda.OutOfMemoryError)
                or "out of memory" in message
                or ("cuda" in message and "memory" in message)
            )
            if not is_cuda_memory_error:
                raise

            self._report(
                progress_callback,
                f"GPU {self.device_index} memory pressure detected; unloading and retrying once…",
                0.46,
            )
            self.unload_model()
            if torch.cuda.is_available():
                with torch.cuda.device(self.device_index):
                    torch.cuda.empty_cache()
            self._load_pipeline(mode, progress_callback)
            with torch.inference_mode():
                result = self.pipe(**self._with_step_progress(kwargs, progress_callback))
            self._report(progress_callback, "Denoising complete; decoding video frames…", 0.91)
            return result


class OptimizedVideoGenerator:
    """Thread-safe pool of per-GPU optimized LTX workers.

    Existing callers can keep using this object like one generator. Each method
    is dispatched to the next available GPU worker and per-request QC state is
    stored in thread-local memory so concurrent Gradio jobs do not overwrite
    each other's reports.
    """

    def __init__(self) -> None:
        visible = torch.cuda.device_count() if torch.cuda.is_available() else 0
        requested = int(os.getenv("LTX_MAX_GPU_WORKERS", str(max(1, visible))))
        worker_count = max(1, min(max(1, visible), max(1, requested)))

        self._workers = [_OptimizedWorker(index) for index in range(worker_count)]
        self._available: queue.Queue[int] = queue.Queue()
        for index in range(worker_count):
            self._available.put(index)
        self._local = threading.local()
        self.worker_count = worker_count

    @property
    def last_qc(self):
        return getattr(self._local, "last_qc", None)

    @property
    def last_skill_plan(self):
        return getattr(self._local, "last_skill_plan", None)

    @property
    def hardware_summary(self) -> str:
        if not torch.cuda.is_available():
            return "CUDA GPU not detected"
        items = []
        for worker in self._workers:
            profile = worker.profile.label if worker.profile else "unknown profile"
            items.append(
                f"GPU {worker.device_index}: {worker.gpu_name} · {worker.vram_total_gb:.1f} GB · {profile}"
            )
        return " | ".join(items)

    def _dispatch(self, method_name: str, *args, **kwargs):
        index = self._available.get()
        worker = self._workers[index]
        try:
            method = getattr(worker, method_name)
            result = method(*args, **kwargs)
            return result
        finally:
            # Snapshot state before another request can reuse this worker.
            self._local.last_qc = worker.last_qc
            self._local.last_skill_plan = worker.last_skill_plan
            self._local.report = worker.generation_report()
            self._available.put(index)

    def generate_text_to_video(self, *args, **kwargs):
        return self._dispatch("generate_text_to_video", *args, **kwargs)

    def generate_image_to_video(self, *args, **kwargs):
        return self._dispatch("generate_image_to_video", *args, **kwargs)

    def generate_conditioned_video(self, *args, **kwargs):
        return self._dispatch("generate_conditioned_video", *args, **kwargs)

    def generation_report(self) -> str:
        return getattr(self._local, "report", "")

    def unload_model(self) -> None:
        # Intended for shutdown/maintenance, not while requests are running.
        for worker in self._workers:
            worker.unload_model()
