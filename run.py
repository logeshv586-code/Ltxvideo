"""One-command bootstrap and launcher: python run.py"""
from __future__ import annotations

import importlib.util
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
    missing = [pkg for module, pkg in REQUIRED.items() if importlib.util.find_spec(module) is None]
    if missing:
        print("Installing missing runtime packages…")
        subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])


def main() -> None:
    ensure_dependencies()
    marker = ROOT / "models" / ".ltx_ready"
    if not marker.exists():
        print("LTX model cache not found. Preparing offline model files…")
        from download_models import download
        download()
    from app import create_app
    app = create_app()
    app.launch(server_name="127.0.0.1", server_port=7860, inbrowser=True, show_error=True)


if __name__ == "__main__":
    main()
