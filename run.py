"""One-command bootstrap and launcher: python run.py"""
from __future__ import annotations

import importlib.util
import os
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
    "psutil": "psutil>=5.9.0",
}


def ensure_dependencies() -> None:
    missing = [pkg for module, pkg in REQUIRED.items() if importlib.util.find_spec(module) is None]
    if missing:
        print("Installing missing runtime packages…")
        subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])


def configure_hardware_profile():
    """Apply GPU-specific budgets before app.py imports the generator constants."""
    import config
    from engine.hardware_profiles import get_active_hardware_profile

    profile = get_active_hardware_profile()
    if profile.key == "no-cuda":
        print("Hardware profile: CUDA GPU not detected")
        return profile

    config.GPU_MEMORY_BUDGET = os.getenv("LTX_GPU_MEMORY_BUDGET", profile.gpu_memory_budget)
    config.CPU_MEMORY_BUDGET = os.getenv("LTX_CPU_MEMORY_BUDGET", profile.cpu_memory_budget)
    config.MAX_NATIVE_FRAMES = min(config.MAX_NATIVE_FRAMES, profile.max_native_frames)

    print(f"Hardware profile: {profile.label}")
    print(
        "Runtime memory: "
        f"GPU {config.GPU_MEMORY_BUDGET} / CPU {config.CPU_MEMORY_BUDGET} / "
        f"native clip cap {config.MAX_NATIVE_FRAMES} frames"
    )
    print(f"Safe first clip: {profile.safe_width}x{profile.safe_height} · {profile.safe_frames} frames")
    return profile


def main() -> int:
    if "--check" in sys.argv[1:]:
        from diagnostics import main as diagnostics_main

        return diagnostics_main()

    ensure_dependencies()
    configure_hardware_profile()

    marker = ROOT / "models" / ".ltx_ready"
    if not marker.exists():
        print("LTX model cache not found. Preparing offline model files…")
        from download_models import download

        download()
    from app import create_app

    app = create_app()
    app.launch(server_name="127.0.0.1", server_port=7860, inbrowser=True, show_error=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
