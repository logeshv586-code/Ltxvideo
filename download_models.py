"""
LTX-2.3 Video Generation Platform — Model Downloader
Downloads model weights from HuggingFace to local storage.
"""

import argparse
import sys
from pathlib import Path

# Monkey-patch httpx to disable SSL verification (fixes Windows cert issues)
import httpx
original_init = httpx.Client.__init__
def new_init(self, *args, **kwargs):
    kwargs['verify'] = False
    original_init(self, *args, **kwargs)
httpx.Client.__init__ = new_init

from huggingface_hub import hf_hub_download, snapshot_download
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))
from config import HF_REPO_ID, GEMMA_DIR, GEMMA_REPO_ID, MODEL_FILES, MODELS_DIR


def download_model_file(repo_id: str, filename: str, local_dir: Path, description: str) -> Path:
    """Download a single model file from HuggingFace."""
    local_path = local_dir / filename
    if local_path.exists():
        size_gb = local_path.stat().st_size / (1024**3)
        print(f"  ✅ Already exists: {filename} ({size_gb:.1f} GB)")
        return local_path

    print(f"  ⬇️  Downloading: {filename}")
    print(f"     {description}")
    downloaded_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
    )
    size_gb = Path(downloaded_path).stat().st_size / (1024**3)
    print(f"  ✅ Complete: {filename} ({size_gb:.1f} GB)")
    return Path(downloaded_path)


def download_gemma_encoder(local_dir: Path) -> Path:
    """Download the Gemma 3 text encoder."""
    if local_dir.exists() and any(local_dir.iterdir()):
        print(f"  ✅ Already exists: Gemma 3 text encoder at {local_dir}")
        return local_dir

    print(f"  ⬇️  Downloading: Gemma 3 4B IT (text encoder)")
    print(f"     This is the text encoder required by LTX-2.3")
    snapshot_download(
        repo_id=GEMMA_REPO_ID,
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
        ignore_patterns=["*.gguf", "*.bin"],  # Skip unnecessary formats
    )
    print(f"  ✅ Complete: Gemma 3 text encoder")
    return local_dir


def check_models() -> dict:
    """Check which models are already downloaded."""
    status = {}
    for key, info in MODEL_FILES.items():
        path = MODELS_DIR / info["filename"]
        status[key] = {
            "exists": path.exists(),
            "path": path,
            "filename": info["filename"],
            "size_gb": path.stat().st_size / (1024**3) if path.exists() else 0,
        }
    # Check Gemma
    status["gemma"] = {
        "exists": GEMMA_DIR.exists() and any(GEMMA_DIR.iterdir()) if GEMMA_DIR.exists() else False,
        "path": GEMMA_DIR,
        "filename": "gemma-3-4b-it/",
    }
    return status


def print_status(status: dict) -> None:
    """Print model download status."""
    print("\n📦 Model Status:")
    print("─" * 60)
    all_ready = True
    for key, info in status.items():
        icon = "✅" if info["exists"] else "❌"
        size = f" ({info.get('size_gb', 0):.1f} GB)" if info.get("size_gb", 0) > 0 else ""
        print(f"  {icon} {info['filename']}{size}")
        if not info["exists"]:
            all_ready = False
    print("─" * 60)
    if all_ready:
        print("  🎉 All models are downloaded and ready!")
    else:
        print("  ⚠️  Some models are missing. Run: python download_models.py")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Download LTX-2.3 model weights")
    parser.add_argument("--check", action="store_true", help="Only check download status")
    parser.add_argument("--skip-gemma", action="store_true", help="Skip Gemma text encoder download")
    args = parser.parse_args()

    print()
    print("╔══════════════════════════════════════════════════╗")
    print("║     LTX-2.3 Model Downloader                    ║")
    print("║     Downloads from HuggingFace                   ║")
    print("╚══════════════════════════════════════════════════╝")
    print()

    if args.check:
        status = check_models()
        print_status(status)
        return

    print(f"📁 Download directory: {MODELS_DIR}")
    print(f"📦 Repository: {HF_REPO_ID}")
    print()

    # Ensure models directory exists
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Download LTX-2.3 model files
    print("━━━ LTX-2.3 Model Files ━━━")
    for key, info in MODEL_FILES.items():
        download_model_file(
            repo_id=HF_REPO_ID,
            filename=info["filename"],
            local_dir=MODELS_DIR,
            description=info["description"],
        )
    print()

    # Download Gemma text encoder
    if not args.skip_gemma:
        print("━━━ Gemma 3 Text Encoder ━━━")
        GEMMA_DIR.mkdir(parents=True, exist_ok=True)
        download_gemma_encoder(GEMMA_DIR)
        print()

    # Final status
    status = check_models()
    print_status(status)


if __name__ == "__main__":
    main()
