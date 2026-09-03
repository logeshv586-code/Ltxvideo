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
    """Apply GPU/VRAM/RAM-specific budgets before importing the video UI."""
    import config
    from engine.hardware_profiles import describe_cuda_hardware, get_active_hardware_profile

    profile = get_active_hardware_profile()
    gpu_count, gpu_descriptions, ram_gb = describe_cuda_hardware()
    if profile.key == "no-cuda":
        print("Hardware profile: CUDA GPU not detected")
        return profile, 1

    config.GPU_MEMORY_BUDGET = os.getenv("LTX_GPU_MEMORY_BUDGET", profile.gpu_memory_budget)
    config.CPU_MEMORY_BUDGET = os.getenv("LTX_CPU_MEMORY_BUDGET", profile.cpu_memory_budget)
    config.MAX_NATIVE_FRAMES = min(config.MAX_NATIVE_FRAMES, profile.max_native_frames)

    requested_workers = int(os.getenv("LTX_MAX_GPU_WORKERS", str(max(1, gpu_count))))
    worker_count = max(1, min(max(1, gpu_count), max(1, requested_workers)))

    print(f"Hardware profile: {profile.label}")
    for description in gpu_descriptions:
        print(f"  {description}")
    print(f"System RAM: {ram_gb:.1f} GB")
    print(
        "Runtime memory: "
        f"GPU budget {config.GPU_MEMORY_BUDGET} per worker / "
        f"CPU budget {config.CPU_MEMORY_BUDGET} / "
        f"native clip cap {config.MAX_NATIVE_FRAMES} frames"
    )
    print(f"Adaptive GPU workers: {worker_count}")
    print(f"Safe first clip: {profile.safe_width}x{profile.safe_height} · {profile.safe_frames} frames")
    if profile.native_scale > 1.0:
        print(
            f"Native quality scaling: {profile.native_scale:.2f}x for normal clips / "
            f"{profile.long_clip_scale:.2f}x for long clips"
        )
    return profile, worker_count


def _arg_value(args: list[str], name: str, default: str) -> str:
    try:
        index = args.index(name)
        return args[index + 1]
    except (ValueError, IndexError):
        return default


def main() -> int:
    args = sys.argv[1:]
    if "--check" in args:
        from diagnostics import main as diagnostics_main

        return diagnostics_main()

    ensure_dependencies()
    is_hunyuan = "--hunyuan-ui" in args
    worker_count = 1

    if is_hunyuan:
        print("Launching Moon Cookie HunyuanVideo-1.5 studio for RTX 4080-class GPUs.")
        print("If setup is incomplete, run: python setup_hunyuan.py --install-code --show-downloads")
        from hunyuan_app import create_app
    else:
        _, worker_count = configure_hardware_profile()
        marker = ROOT / "models" / ".ltx_ready"
        if not marker.exists():
            print("LTX model cache not found. Preparing offline model files…")
            from download_models import download

            download()

        if "--legacy-ui" in args:
            print("Launching legacy multi-studio UI (--legacy-ui).")
            # Legacy UI owns one generator instance, so keep its queue serial.
            worker_count = 1
            from app import create_app
        else:
            print("Launching Easy Video Creator with adaptive GPU workers.")
            from easy_app import create_app

    app = create_app()
    app.queue(default_concurrency_limit=worker_count, max_size=max(8, worker_count * 4))

    server_mode = "--server" in args
    default_host = "0.0.0.0" if server_mode else "127.0.0.1"
    server_name = os.getenv("LTX_SERVER_NAME", default_host)
    server_port = int(_arg_value(args, "--port", os.getenv("LTX_SERVER_PORT", "7860")))
    inbrowser = not server_mode and server_name in {"127.0.0.1", "localhost"}

    if server_name == "0.0.0.0":
        print(f"Server UI: http://<SERVER-IP>:{server_port}")
        print("Use a firewall/VPN/reverse proxy if this machine is reachable from the public internet.")
    else:
        print(f"UI: http://{server_name}:{server_port}")

    app.launch(
        server_name=server_name,
        server_port=server_port,
        inbrowser=inbrowser,
        show_error=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
