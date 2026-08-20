"""One-command bootstrap and launcher: python run.py"""

from __future__ import annotations

import argparse
import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.resolve()

REQUIRED = {
    "gradio": "gradio>=5.0.0",
    "torch": "torch>=2.4.0",
    "diffusers": "diffusers>=0.37.1",
    "transformers": "transformers>=4.48.0",
    "accelerate": "accelerate>=1.2.0",
    "bitsandbytes": "bitsandbytes>=0.45.0",
    "cv2": "opencv-python-headless>=4.8.0",
    "imageio_ffmpeg": "imageio-ffmpeg>=0.5.1",
}


def ensure_dependencies() -> None:
    missing = [
        pkg
        for module, pkg in REQUIRED.items()
        if importlib.util.find_spec(module) is None
    ]

    if missing:
        print("Installing missing runtime packages…")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", *missing]
        )


def print_check(status: str, message: str) -> None:
    print(f"[{status}] {message}")


def run_diagnostics() -> int:
    print("LTX Video Diagnostics")
    print("---------------------")

    blocking_failure = False

    # Python version
    python_version = sys.version_info
    python_display = (
        f"{python_version.major}.{python_version.minor}.{python_version.micro}"
    )

    if python_version >= (3, 10):
        print_check("PASS", f"Python {python_display} (3.10+)")
    else:
        print_check(
            "WARN",
            f"Python {python_display} detected; Python 3.10+ is required",
        )
        blocking_failure = True

    # PyTorch import
    torch = None

    try:
        import torch

        print_check("PASS", f"PyTorch {torch.__version__} imported")
    except Exception as exc:
        print_check("WARN", f"PyTorch import failed: {exc}")
        blocking_failure = True

    # CUDA / GPU
    if torch is not None:
        try:
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                print_check("PASS", "CUDA available")
                print_check("PASS", f"GPU: {gpu_name}")
            else:
                print_check("WARN", "CUDA unavailable")
        except Exception as exc:
            print_check("WARN", f"CUDA check failed: {exc}")

    # FFmpeg
    if shutil.which("ffmpeg") is not None:
        print_check("PASS", "FFmpeg available on PATH")
    else:
        try:
            import imageio_ffmpeg

            ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

            if Path(ffmpeg_exe).exists():
                print_check("PASS", "FFmpeg available through imageio-ffmpeg")
            else:
                print_check("WARN", "FFmpeg executable was not found")
        except Exception as exc:
            print_check("WARN", f"FFmpeg check failed: {exc}")

    # LTX model cache
    marker = ROOT / "models" / ".ltx_ready"
    cache_dir = ROOT / "models" / "hf-cache"

    if marker.exists():
        print_check("PASS", f"LTX model-ready marker found: {marker}")
    elif cache_dir.exists():
        print_check(
            "WARN",
            "LTX model cache directory exists, but .ltx_ready is missing",
        )
    else:
        print_check("WARN", "LTX model cache not found")

    print()

    if blocking_failure:
        print("Diagnostics finished with blocking problems.")
        return 1

    print("Diagnostics complete.")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check",
        action="store_true",
        help="Run startup diagnostics without installing packages or downloading the model",
    )
    args = parser.parse_args()

    if args.check:
        raise SystemExit(run_diagnostics())

    ensure_dependencies()

    marker = ROOT / "models" / ".ltx_ready"

    if not marker.exists():
        print("LTX model cache not found. Preparing offline model files…")

        from download_models import download

        download()

    from app import create_app

    app = create_app()

    app.launch(
        server_name="127.0.0.1",
        server_port=7860,
        inbrowser=True,
        show_error=True,
    )


if __name__ == "__main__":
    main()