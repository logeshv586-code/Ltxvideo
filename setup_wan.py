"""Explicit setup for Wan2.1-T2V-1.3B.

For RTX 4050 / 6 GB VRAM, ``--download-gguf`` is the recommended path. It
downloads the official Diffusers components without the full transformer and
adds a Q5_0 GGUF transformer for lower memory use.
"""
from __future__ import annotations

import argparse

from engine.video_enhancement import enhancement_status
from engine.wan_generator import (
    WAN_GGUF_BASE_DIR,
    WAN_GGUF_DIR,
    WAN_GGUF_FILENAME,
    WAN_GGUF_PATH,
    WAN_GGUF_REPO_ID,
    WAN_MODEL_DIR,
    WAN_REPO_ID,
)


def _print_enhancement_status() -> None:
    status = enhancement_status()
    print("Wan post-processing:")
    print(f"  RIFE: {status['rife']}")
    print(f"  Real-ESRGAN: {status['realesrgan']}")
    print("Optional overrides: RIFE_EXE, REALESRGAN_EXE, WAN_SMOOTHING, WAN_UPSCALER")


def _print_model_status() -> None:
    print("Wan model setup:")
    print(f"  Full Diffusers path: {WAN_MODEL_DIR}")
    print(f"  GGUF base components: {WAN_GGUF_BASE_DIR}")
    print(f"  GGUF transformer: {WAN_GGUF_PATH}")
    print(f"  GGUF transformer present: {'yes' if WAN_GGUF_PATH.exists() else 'no'}")
    print("  RTX 4050 recommendation: WAN_BACKEND=gguf + WAN_OFFLOAD_MODE=sequential")


def _download_full() -> None:
    from huggingface_hub import snapshot_download

    print(f"Downloading full {WAN_REPO_ID} to {WAN_MODEL_DIR} ...")
    snapshot_download(repo_id=WAN_REPO_ID, local_dir=WAN_MODEL_DIR)
    print("Full Wan2.1 Diffusers download complete.")


def _download_gguf() -> None:
    from huggingface_hub import hf_hub_download, snapshot_download

    print("Preparing low-memory Wan2.1 GGUF runtime...")
    print(f"Official pipeline components: {WAN_REPO_ID}")
    print(f"GGUF transformer: {WAN_GGUF_REPO_ID}/{WAN_GGUF_FILENAME}")

    # Keep the transformer config but skip its multi-gigabyte safetensor
    # weights. The quantized GGUF file replaces those weights at runtime.
    snapshot_download(
        repo_id=WAN_REPO_ID,
        local_dir=WAN_GGUF_BASE_DIR,
        ignore_patterns=[
            "transformer/*.safetensors",
            "transformer/*.safetensors.index.json",
            "transformer/diffusion_pytorch_model*.bin",
        ],
    )
    WAN_GGUF_DIR.mkdir(parents=True, exist_ok=True)
    hf_hub_download(
        repo_id=WAN_GGUF_REPO_ID,
        filename=WAN_GGUF_FILENAME,
        local_dir=WAN_GGUF_DIR,
    )
    print(f"GGUF setup complete: {WAN_GGUF_PATH}")
    print("Recommended for RTX 4050 6 GB:")
    print("  WAN_BACKEND=gguf")
    print("  WAN_OFFLOAD_MODE=sequential")
    print("  WAN_DELIVERY_FPS=32")


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare Wan2.1-T2V-1.3B for the video studio.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--download-gguf",
        action="store_true",
        help="Recommended for 4-8 GB GPUs: download Q5_0 GGUF transformer plus official pipeline components.",
    )
    group.add_argument(
        "--download",
        action="store_true",
        help="Download the full Diffusers checkpoint (better suited to higher-VRAM GPUs).",
    )
    parser.add_argument("--status", action="store_true", help="Show Wan model and enhancement setup status.")
    args = parser.parse_args()

    if args.status:
        _print_model_status()
        _print_enhancement_status()
        return 0

    if args.download_gguf:
        _download_gguf()
        _print_enhancement_status()
        print("Launch with: python run.py --video-studio-ui")
        return 0

    if args.download:
        _download_full()
        _print_enhancement_status()
        print("Launch with: python run.py --video-studio-ui")
        return 0

    _print_model_status()
    print()
    print("RTX 4050 / 6 GB VRAM quick setup:")
    print("  python setup_wan.py --download-gguf")
    print()
    print("Higher-VRAM full-model setup:")
    print("  python setup_wan.py --download")
    print()
    _print_enhancement_status()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
