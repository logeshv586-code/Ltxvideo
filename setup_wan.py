"""Explicit Wan2.1-T2V-1.3B setup; downloads never occur during app startup."""
from __future__ import annotations

import argparse

from engine.video_enhancement import enhancement_status
from engine.wan_generator import WAN_MODEL_DIR, WAN_REPO_ID


def _print_enhancement_status() -> None:
    status = enhancement_status()
    print("Wan post-processing:")
    print(f"  RIFE: {status['rife']}")
    print(f"  Real-ESRGAN: {status['realesrgan']}")
    print("Optional overrides: RIFE_EXE, REALESRGAN_EXE, WAN_SMOOTHING, WAN_UPSCALER")


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare Wan2.1-T2V-1.3B for the video studio.")
    parser.add_argument("--download", action="store_true", help="Download the Wan2.1-T2V-1.3B checkpoint.")
    parser.add_argument("--status", action="store_true", help="Show Wan model and enhancement setup status.")
    args = parser.parse_args()

    if args.status:
        print(f"Wan2.1 model directory: {WAN_MODEL_DIR}")
        _print_enhancement_status()
        return 0

    if not args.download:
        print(f"Wan2.1-T2V-1.3B will be stored in: {WAN_MODEL_DIR}")
        print("Start the explicit download with: python setup_wan.py --download")
        _print_enhancement_status()
        return 0

    from huggingface_hub import snapshot_download

    print(f"Downloading {WAN_REPO_ID} to {WAN_MODEL_DIR} …")
    snapshot_download(repo_id=WAN_REPO_ID, local_dir=WAN_MODEL_DIR)
    print("Wan2.1 download complete.")
    _print_enhancement_status()
    print("Launch with: python run.py --video-studio-ui --server")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
