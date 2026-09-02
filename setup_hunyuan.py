"""Install the official HunyuanVideo-1.5 source and show checkpoint commands.

Examples:
  python setup_hunyuan.py --install-code
  python setup_hunyuan.py --show-downloads

Checkpoint downloads are intentionally explicit because the vision encoder
requires the user's own Hugging Face access approval/token for FLUX.1-Redux-dev.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from config import MODELS_DIR, PROJECT_ROOT

SOURCE_DIR = PROJECT_ROOT / "third_party" / "HunyuanVideo-1.5"
MODEL_DIR = MODELS_DIR / "hunyuanvideo-1.5"
OFFICIAL_REPO = "https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5.git"


def install_code() -> None:
    SOURCE_DIR.parent.mkdir(parents=True, exist_ok=True)
    if not SOURCE_DIR.exists():
        subprocess.check_call(["git", "clone", "--depth", "1", OFFICIAL_REPO, str(SOURCE_DIR)])
    else:
        print(f"Official source already exists: {SOURCE_DIR}")
    requirements = SOURCE_DIR / "requirements.txt"
    if requirements.exists():
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", str(requirements)])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-U", "huggingface_hub[cli]", "modelscope"])
    print("Official Hunyuan source/runtime installed.")


def show_downloads() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    print("\nRun these commands from the Ltxvideo repository root:\n")
    print(f'hf download tencent/HunyuanVideo-1.5 --local-dir "{MODEL_DIR}"')
    print(f'hf download Qwen/Qwen2.5-VL-7B-Instruct --local-dir "{MODEL_DIR / "text_encoder" / "llm"}"')
    print(f'hf download google/byt5-small --local-dir "{MODEL_DIR / "text_encoder" / "byt5-small"}"')
    print(f'modelscope download --model AI-ModelScope/Glyph-SDXL-v2 --local_dir "{MODEL_DIR / "text_encoder" / "Glyph-SDXL-v2"}"')
    print("\nThe SigLIP vision encoder comes from FLUX.1-Redux-dev and requires approved Hugging Face access.")
    print("Replace <YOUR_HF_TOKEN> with your own token after access is approved:")
    print(f'hf download black-forest-labs/FLUX.1-Redux-dev --local-dir "{MODEL_DIR / "vision_encoder" / "siglip"}" --token <YOUR_HF_TOKEN>')
    print("\nAfter downloads finish, launch: python run.py --hunyuan-ui")


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare HunyuanVideo-1.5 for the LTX Video Studio")
    parser.add_argument("--install-code", action="store_true", help="Clone official source and install its Python requirements")
    parser.add_argument("--show-downloads", action="store_true", help="Print official checkpoint download commands")
    args = parser.parse_args()
    if not args.install_code and not args.show_downloads:
        parser.print_help()
        return 0
    if args.install_code:
        install_code()
    if args.show_downloads:
        show_downloads()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
