"""Install the CogVideoX-5B checkpoint explicitly, never during app startup."""
from __future__ import annotations

import argparse

from engine.cogvideox_generator import COGVIDEOX_MODEL_DIR, COGVIDEOX_REPO_ID


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare CogVideoX-5B for the Quality Clip Studio.")
    parser.add_argument("--download", action="store_true", help="Download the roughly 22 GB CogVideoX-5B checkpoint.")
    args = parser.parse_args()
    if not args.download:
        print(f"CogVideoX-5B will be stored in: {COGVIDEOX_MODEL_DIR}")
        print("The checkpoint is about 22 GB. Start the explicit download with:")
        print("  python setup_cogvideox.py --download")
        return 0

    from huggingface_hub import snapshot_download

    print(f"Downloading {COGVIDEOX_REPO_ID} to {COGVIDEOX_MODEL_DIR} …")
    snapshot_download(repo_id=COGVIDEOX_REPO_ID, local_dir=COGVIDEOX_MODEL_DIR)
    print("CogVideoX-5B download complete. Launch with: python run.py --cogvideox-ui --server")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
