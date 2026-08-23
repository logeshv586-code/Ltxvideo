"""Lightweight startup diagnostics that never downloads or loads the LTX model."""
from __future__ import annotations

import importlib.util
import platform
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from engine.hardware_profiles import select_hardware_profile

ROOT = Path(__file__).parent.resolve()
MODEL_MARKER = ROOT / "models" / ".ltx_ready"
MIN_PYTHON = (3, 10)


@dataclass(frozen=True)
class CheckResult:
    level: str
    name: str
    detail: str


Loader = Callable[[str], object]


def _load_module(name: str) -> object:
    return __import__(name)


def collect_diagnostics(*, loader: Loader = _load_module, marker: Path = MODEL_MARKER) -> list[CheckResult]:
    results: list[CheckResult] = []

    py_ok = sys.version_info >= MIN_PYTHON
    results.append(
        CheckResult(
            "PASS" if py_ok else "FAIL",
            "Python",
            f"{platform.python_version()} (requires 3.10+)",
        )
    )

    torch_spec = importlib.util.find_spec("torch")
    if torch_spec is None:
        results.append(CheckResult("FAIL", "PyTorch", "not installed"))
        results.append(CheckResult("WARN", "CUDA", "cannot check CUDA until PyTorch is installed"))
    else:
        try:
            torch = loader("torch")
            version = getattr(torch, "__version__", "unknown")
            results.append(CheckResult("PASS", "PyTorch", str(version)))
            cuda = getattr(torch, "cuda", None)
            cuda_available = bool(cuda and cuda.is_available())
            if cuda_available:
                gpu_name = cuda.get_device_name(0)
                props = cuda.get_device_properties(0)
                vram_total_gb = props.total_memory / 1024**3
                cuda_version = getattr(getattr(torch, "version", None), "cuda", None) or "unknown"
                results.append(CheckResult("PASS", "CUDA", f"{gpu_name} · {vram_total_gb:.1f} GB VRAM · CUDA {cuda_version}"))
                profile = select_hardware_profile(gpu_name, vram_total_gb)
                results.append(
                    CheckResult(
                        "PASS",
                        "GPU profile",
                        f"{profile.label} · GPU {profile.gpu_memory_budget} / CPU {profile.cpu_memory_budget}",
                    )
                )
                results.append(
                    CheckResult(
                        "INFO",
                        "Safe preset",
                        f"{profile.safe_width}x{profile.safe_height} · {profile.safe_frames} frames · native cap {profile.max_native_frames}",
                    )
                )
            else:
                results.append(CheckResult("WARN", "CUDA", "no CUDA GPU detected; generation will not run on the target GPU path"))
        except Exception as exc:  # keep diagnostics usable even with a broken torch install
            results.append(CheckResult("FAIL", "PyTorch", f"import failed: {exc}"))
            results.append(CheckResult("WARN", "CUDA", "not checked because PyTorch import failed"))

    ffmpeg_detail = None
    if importlib.util.find_spec("imageio_ffmpeg") is not None:
        try:
            module = loader("imageio_ffmpeg")
            get_exe = getattr(module, "get_ffmpeg_exe", None)
            if callable(get_exe):
                ffmpeg_detail = str(get_exe())
        except Exception:
            ffmpeg_detail = None
    if not ffmpeg_detail:
        ffmpeg_detail = shutil.which("ffmpeg")
    results.append(
        CheckResult(
            "PASS" if ffmpeg_detail else "WARN",
            "FFmpeg",
            ffmpeg_detail or "not found; delivery export/story stitching may be unavailable",
        )
    )

    results.append(
        CheckResult(
            "PASS" if marker.exists() else "INFO",
            "Model cache",
            str(marker) if marker.exists() else "not prepared yet; first normal launch will download/cache the model",
        )
    )
    return results


def exit_code(results: list[CheckResult]) -> int:
    return 1 if any(result.level == "FAIL" for result in results) else 0


def print_report(results: list[CheckResult]) -> None:
    print("Ltxvideo startup diagnostics")
    print("=" * 28)
    for result in results:
        print(f"[{result.level:<4}] {result.name}: {result.detail}")
    print()
    safe = next((result.detail for result in results if result.name == "Safe preset"), None)
    if safe:
        print(f"Recommended first generation: {safe}")
    else:
        print("Safe first generation: 384x224 · 49 frames (~1.6 s at 30 FPS)")


def main() -> int:
    results = collect_diagnostics()
    print_report(results)
    return exit_code(results)


if __name__ == "__main__":
    raise SystemExit(main())
