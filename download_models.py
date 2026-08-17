"""Download the exact LTX-2.3 assets required by this app."""

import argparse
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

from config import GEMMA_DIR, GEMMA_REPO_ID, HF_REPO_ID, MODEL_FILES, MODELS_DIR


def download_model_file(repo_id: str, filename: str, local_dir: Path, description: str) -> Path:
    local_path = local_dir / filename
    if local_path.exists() and local_path.stat().st_size > 0:
        print(f"✅ {filename} already exists ({local_path.stat().st_size / 1024**3:.1f} GB)")
        return local_path

    print(f"⬇️  {description}: {filename}")
    downloaded = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=str(local_dir),
    )
    path = Path(downloaded)
    print(f"✅ Downloaded {filename} ({path.stat().st_size / 1024**3:.1f} GB)")
    return path


def download_gemma_encoder() -> Path:
    if GEMMA_DIR.exists() and any(GEMMA_DIR.iterdir()):
        print(f"✅ Gemma encoder already exists: {GEMMA_DIR}")
        return GEMMA_DIR

    print(f"⬇️  Downloading official LTX-2.3 text encoder: {GEMMA_REPO_ID}")
    print("   If Hugging Face returns 401/403, accept the model terms and run `hf auth login` first.")
    GEMMA_DIR.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=GEMMA_REPO_ID,
        local_dir=str(GEMMA_DIR),
        ignore_patterns=["*.gguf"],
    )
    print("✅ Gemma text encoder downloaded.")
    return GEMMA_DIR


def check_models() -> dict:
    status = {}
    for key, info in MODEL_FILES.items():
        path = MODELS_DIR / info["filename"]
        status[key] = {
            "exists": path.exists() and path.stat().st_size > 0,
            "path": path,
            "filename": info["filename"],
            "size_gb": path.stat().st_size / 1024**3 if path.exists() else 0,
        }
    status["gemma"] = {
        "exists": GEMMA_DIR.exists() and any(GEMMA_DIR.iterdir()),
        "path": GEMMA_DIR,
        "filename": GEMMA_DIR.name + "/",
        "size_gb": 0,
    }
    return status


def print_status(status: dict) -> bool:
    print("\n📦 Model status")
    print("─" * 70)
    ready = True
    for info in status.values():
        icon = "✅" if info["exists"] else "❌"
        size = f" ({info['size_gb']:.1f} GB)" if info.get("size_gb") else ""
        print(f"{icon} {info['filename']}{size}")
        ready = ready and info["exists"]
    print("─" * 70)
    print("🎉 Ready for generation." if ready else "⚠️ Missing assets. Run: python download_models.py")
    return ready


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="Only verify local assets")
    parser.add_argument("--skip-gemma", action="store_true")
    args = parser.parse_args()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    if args.check:
        print_status(check_models())
        return

    for info in MODEL_FILES.values():
        download_model_file(HF_REPO_ID, info["filename"], MODELS_DIR, info["description"])

    if not args.skip_gemma:
        download_gemma_encoder()

    print_status(check_models())


if __name__ == "__main__":
    main()
