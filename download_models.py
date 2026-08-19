"""Pre-download the official LTX Diffusers files for offline reuse."""
from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download

from config import HF_REPO_ID, MODELS_DIR


def download() -> Path:
    print("Downloading official Lightricks/LTX-Video 2B Diffusers assets…")
    print("This is a one-time download. Hugging Face cache is reused afterwards.")
    path = snapshot_download(
        repo_id=HF_REPO_ID,
        cache_dir=str(MODELS_DIR / "hf-cache"),
        allow_patterns=[
            "model_index.json",
            "scheduler/*",
            "tokenizer/*",
            "transformer/*",
            "text_encoder/*",
            "vae/*",
        ],
        resume_download=True,
    )
    marker = MODELS_DIR / ".ltx_ready"
    marker.write_text(str(path), encoding="utf-8")
    print(f"Ready: {path}")
    return Path(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="Show cache folder only")
    args = parser.parse_args()
    cache = MODELS_DIR / "hf-cache"
    if args.check:
        print(cache)
        return
    download()


if __name__ == "__main__":
    main()
