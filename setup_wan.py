"""Explicit Wan2.1-T2V-1.3B setup; downloads never occur during app startup."""
from __future__ import annotations

import argparse

from engine.wan_generator import WAN_MODEL_DIR, WAN_REPO_ID


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare Wan2.1-T2V-1.3B for the video studio.")
    parser.add_argument("--download", action="store_true", help="Download the Wan2.1-T2V-1.3B checkpoint.")
    args = parser.parse_args()
    if not args.download:
        print(f"Wan2.1-T2V-1.3B will be stored in: {WAN_MODEL_DIR}")
        print("Start the explicit download with: python setup_wan.py --download")
        return 0
    from huggingface_hub import snapshot_download

    print(f"Downloading {WAN_REPO_ID} to {WAN_MODEL_DIR} …")
    snapshot_download(repo_id=WAN_REPO_ID, local_dir=WAN_MODEL_DIR)
    print("Wan2.1 download complete. Launch with: python run.py --video-studio-ui --server")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
