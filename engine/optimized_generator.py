"""RTX 40-series execution wrapper for the existing low-memory LTX generator.

The base generator is intentionally kept as the compatibility path. This
wrapper removes a costly CUDA cache flush from every successful inference,
enables inference-only execution, and keeps cache clearing only for actual OOM
recovery. That matters for sequential long-form generation where dozens of
short clips are produced back-to-back.
"""
from __future__ import annotations

import torch

from engine.generator import Progress, VideoGenerator


class OptimizedVideoGenerator(VideoGenerator):
    """Faster steady-state execution for RTX 3050/4050/4060-class Ada GPUs."""

    def __init__(self) -> None:
        super().__init__()
        if torch.cuda.is_available():
            try:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                torch.backends.cudnn.benchmark = True
                torch.set_float32_matmul_precision("high")
            except Exception:
                # These flags are performance hints; generation must still work
                # on PyTorch/CUDA builds that do not expose every option.
                pass

    def _run_pipe_with_cuda_retry(self, mode: str, kwargs: dict, progress_callback: Progress):
        """Run without empty_cache() on the normal path; clear only after OOM."""
        self._load_pipeline(mode, progress_callback)
        try:
            with torch.inference_mode():
                return self.pipe(**kwargs)
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
                "CUDA memory pressure detected; unloading, clearing cache and retrying once…",
                0.46,
            )
            self.unload_model()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self._load_pipeline(mode, progress_callback)
            with torch.inference_mode():
                return self.pipe(**kwargs)
