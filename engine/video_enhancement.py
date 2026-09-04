"""Optional low-VRAM post-processing for Wan2.1 cartoon videos.

The Wan generator stays at its stable 832x480 / 16 FPS native shape. This
module improves delivery quality after diffusion has finished:

1. RIFE interpolation (when ``rife-ncnn-vulkan`` is available) to 32 FPS.
2. FFmpeg motion-compensated interpolation as a zero-setup RIFE fallback.
3. Real-ESRGAN anime/video upscaling (when ``realesrgan-ncnn-vulkan`` is
   available) before the normal 1080p delivery encode.
4. Lanczos + light sharpening in ``export_delivery`` when Real-ESRGAN is not
   installed.

The NCNN/Vulkan executables are intentionally optional. They do not add a
second PyTorch model to the Wan CUDA process, which is important on a 6 GB RTX
4050 laptop.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Callable

from engine.video_processor import _ffmpeg_exe, export_delivery, get_video_info

StatusCallback = Callable[[str], None] | None

DEFAULT_REALESRGAN_MODEL = "realesr-animevideov3"
DEFAULT_REALESRGAN_SCALE = 2
DEFAULT_REALESRGAN_TILE = 256


def _status(callback: StatusCallback, message: str) -> None:
    if callback:
        callback(message)


def _find_executable(env_name: str, candidates: tuple[str, ...]) -> str | None:
    configured = os.getenv(env_name, "").strip().strip('"')
    if configured:
        configured_path = Path(configured).expanduser()
        if configured_path.exists():
            return str(configured_path.resolve())
    for candidate in candidates:
        found = shutil.which(candidate)
        if found:
            return found
    return None


def find_realesrgan_executable() -> str | None:
    return _find_executable(
        "REALESRGAN_EXE",
        ("realesrgan-ncnn-vulkan", "realesrgan-ncnn-vulkan.exe"),
    )


def find_rife_executable() -> str | None:
    return _find_executable(
        "RIFE_EXE",
        ("rife-ncnn-vulkan", "rife-ncnn-vulkan.exe"),
    )


def enhancement_status() -> dict[str, str]:
    """Return lightweight setup information without loading any AI model."""
    realesrgan = find_realesrgan_executable()
    rife = find_rife_executable()
    return {
        "realesrgan": realesrgan or "not installed; Lanczos fallback will be used",
        "rife": rife or "not installed; FFmpeg motion interpolation fallback will be used",
    }


def _run(cmd: list[str], label: str) -> None:
    # NCNN release bundles keep their model folders beside the executable.
    # Running from that directory means their default relative model paths work
    # even when Ltxvideo itself was launched from another folder.
    executable = Path(cmd[0]).expanduser()
    cwd = str(executable.resolve().parent) if executable.exists() else None
    completed = subprocess.run(
        cmd,
        cwd=cwd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        errors="replace",
    )
    if completed.returncode != 0:
        detail = (completed.stderr or "").strip().splitlines()
        tail = " | ".join(detail[-6:]) if detail else "no stderr output"
        raise RuntimeError(f"{label} failed ({completed.returncode}): {tail}")


def _extract_frames(video_path: str | Path, frame_dir: Path) -> list[Path]:
    frame_dir.mkdir(parents=True, exist_ok=True)
    pattern = frame_dir / "%08d.png"
    _run(
        [
            _ffmpeg_exe(),
            "-y",
            "-i",
            str(Path(video_path).resolve()),
            "-vsync",
            "0",
            "-start_number",
            "0",
            str(pattern),
        ],
        "FFmpeg frame extraction",
    )
    frames = sorted(frame_dir.glob("*.png"))
    if not frames:
        raise RuntimeError(f"No frames were extracted from {video_path}")
    return frames


def _encode_frames(frame_dir: Path, output_path: str | Path, fps: float) -> Path:
    output = Path(output_path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    _run(
        [
            _ffmpeg_exe(),
            "-y",
            "-framerate",
            f"{float(fps):.6f}",
            "-start_number",
            "0",
            "-i",
            str(frame_dir / "%08d.png"),
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "slow",
            "-crf",
            "14",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output),
        ],
        "FFmpeg frame encode",
    )
    return output


def interpolate_with_rife(
    video_path: str | Path,
    output_path: str | Path,
    target_fps: int,
    executable: str | None = None,
    gpu_id: int = 0,
    callback: StatusCallback = None,
) -> Path:
    """Interpolate the whole frame directory with the portable RIFE binary."""
    exe = executable or find_rife_executable()
    if not exe:
        raise RuntimeError("rife-ncnn-vulkan was not found")

    info = get_video_info(video_path)
    source_fps = float(info.get("fps") or 0.0)
    if source_fps <= 0:
        raise RuntimeError("Cannot determine source FPS for RIFE interpolation")
    multiplier = max(1, int(round(float(target_fps) / source_fps)))
    if multiplier != 2:
        raise ValueError("This low-VRAM RIFE path currently supports 2x FPS interpolation only")

    with tempfile.TemporaryDirectory(prefix="ltx_rife_") as td:
        root = Path(td)
        source_dir = root / "source"
        smooth_dir = root / "smooth"
        smooth_dir.mkdir(parents=True, exist_ok=True)
        frames = _extract_frames(video_path, source_dir)
        if len(frames) < 2:
            raise RuntimeError("RIFE needs at least two source frames")

        _status(callback, f"RIFE processing {len(frames)} source frames in one batch")
        _run(
            [
                exe,
                "-i",
                str(source_dir),
                "-o",
                str(smooth_dir),
                "-n",
                str(len(frames) * 2),
                "-g",
                str(int(gpu_id)),
                "-f",
                "%08d.png",
            ],
            "RIFE frame interpolation",
        )
        output_frames = sorted(smooth_dir.glob("*.png"))
        if len(output_frames) < len(frames):
            raise RuntimeError("RIFE did not produce the expected output frame sequence")
        _status(callback, f"RIFE created {len(output_frames)} smooth frames")
        return _encode_frames(smooth_dir, output_path, float(target_fps))


def interpolate_with_ffmpeg(
    video_path: str | Path,
    output_path: str | Path,
    target_fps: int,
) -> Path:
    """Motion-compensated CPU fallback when RIFE is not installed."""
    output = Path(output_path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    filter_expr = (
        f"minterpolate=fps={int(target_fps)}:mi_mode=mci:mc_mode=aobmc:"
        "me_mode=bidir:vsbmc=1"
    )
    _run(
        [
            _ffmpeg_exe(),
            "-y",
            "-i",
            str(Path(video_path).resolve()),
            "-vf",
            filter_expr,
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "slow",
            "-crf",
            "14",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output),
        ],
        "FFmpeg motion interpolation",
    )
    return output


def upscale_with_realesrgan(
    video_path: str | Path,
    output_path: str | Path,
    scale: int = DEFAULT_REALESRGAN_SCALE,
    model: str = DEFAULT_REALESRGAN_MODEL,
    tile: int = DEFAULT_REALESRGAN_TILE,
    executable: str | None = None,
    gpu_id: int = 0,
    callback: StatusCallback = None,
) -> Path:
    """Upscale a video frame directory with Real-ESRGAN NCNN/Vulkan."""
    exe = executable or find_realesrgan_executable()
    if not exe:
        raise RuntimeError("realesrgan-ncnn-vulkan was not found")

    info = get_video_info(video_path)
    source_fps = float(info.get("fps") or 0.0)
    if source_fps <= 0:
        raise RuntimeError("Cannot determine source FPS for Real-ESRGAN")

    scale = max(2, min(4, int(scale)))
    tile = max(0, int(tile))
    with tempfile.TemporaryDirectory(prefix="ltx_realesrgan_") as td:
        root = Path(td)
        source_dir = root / "source"
        upscale_dir = root / "upscale"
        upscale_dir.mkdir(parents=True, exist_ok=True)
        frames = _extract_frames(video_path, source_dir)

        cmd = [
            exe,
            "-i",
            str(source_dir),
            "-o",
            str(upscale_dir),
            "-n",
            model,
            "-s",
            str(scale),
            "-f",
            "png",
            "-g",
            str(int(gpu_id)),
        ]
        if tile > 0:
            cmd.extend(["-t", str(tile)])
        _status(callback, f"Real-ESRGAN processing {len(frames)} frames in one batch")
        _run(cmd, "Real-ESRGAN video upscaling")
        output_frames = sorted(upscale_dir.glob("*.png"))
        if len(output_frames) != len(frames):
            raise RuntimeError(
                f"Real-ESRGAN returned {len(output_frames)} frames for {len(frames)} inputs"
            )
        _status(callback, f"Real-ESRGAN enhanced {len(output_frames)} frames")
        return _encode_frames(upscale_dir, output_path, source_fps)


def enhance_wan_delivery(
    video_path: str | Path,
    output_path: str | Path,
    width: int = 1920,
    height: int = 1080,
    native_fps: int = 16,
    target_fps: int = 32,
    gpu_id: int = 0,
    callback: StatusCallback = None,
) -> Path:
    """Apply the best available smoothing/upscaling path, then encode delivery.

    Environment controls:

    ``WAN_SMOOTHING``: ``auto`` (default), ``rife``, ``ffmpeg`` or ``off``.
    ``WAN_UPSCALER``: ``auto`` (default), ``realesrgan`` or ``off``.
    ``RIFE_EXE`` / ``REALESRGAN_EXE``: optional explicit executable paths.
    ``REALESRGAN_MODEL``: defaults to ``realesr-animevideov3`` for cartoons.
    ``REALESRGAN_SCALE``: defaults to 2.
    ``REALESRGAN_TILE``: defaults to 256 for low-VRAM laptops.
    """
    source = Path(video_path).resolve()
    output = Path(output_path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    smoothing = os.getenv("WAN_SMOOTHING", "auto").strip().lower()
    upscaler = os.getenv("WAN_UPSCALER", "auto").strip().lower()
    if smoothing not in {"auto", "rife", "ffmpeg", "off"}:
        smoothing = "auto"
    if upscaler not in {"auto", "realesrgan", "off"}:
        upscaler = "auto"

    actual_fps = int(native_fps)
    with tempfile.TemporaryDirectory(prefix="ltx_wan_enhance_") as td:
        root = Path(td)
        current = source

        if smoothing != "off" and int(target_fps) > int(native_fps):
            smoothed = root / "wan_smoothed.mp4"
            rife = find_rife_executable()
            used_rife = False
            if smoothing in {"auto", "rife"} and rife:
                try:
                    _status(callback, f"Smoothing Wan motion with RIFE: {native_fps} → {target_fps} FPS")
                    interpolate_with_rife(
                        current,
                        smoothed,
                        target_fps,
                        executable=rife,
                        gpu_id=gpu_id,
                        callback=callback,
                    )
                    current = smoothed
                    actual_fps = int(target_fps)
                    used_rife = True
                except Exception as exc:
                    _status(callback, f"RIFE unavailable for this render ({exc}); using FFmpeg motion interpolation")

            if not used_rife and smoothing in {"auto", "rife", "ffmpeg"}:
                try:
                    _status(callback, f"Smoothing Wan motion with FFmpeg: {native_fps} → {target_fps} FPS")
                    interpolate_with_ffmpeg(current, smoothed, target_fps)
                    current = smoothed
                    actual_fps = int(target_fps)
                except Exception as exc:
                    actual_fps = int(native_fps)
                    _status(callback, f"Motion interpolation skipped ({exc}); keeping native {native_fps} FPS")

        if upscaler != "off":
            realesrgan = find_realesrgan_executable()
            if upscaler in {"auto", "realesrgan"} and realesrgan:
                upscaled = root / "wan_realesrgan.mp4"
                model = os.getenv("REALESRGAN_MODEL", DEFAULT_REALESRGAN_MODEL).strip() or DEFAULT_REALESRGAN_MODEL
                scale = int(os.getenv("REALESRGAN_SCALE", str(DEFAULT_REALESRGAN_SCALE)))
                tile = int(os.getenv("REALESRGAN_TILE", str(DEFAULT_REALESRGAN_TILE)))
                try:
                    _status(callback, f"Upscaling Wan cartoon frames with Real-ESRGAN {model} ({scale}x)")
                    upscale_with_realesrgan(
                        current,
                        upscaled,
                        scale=scale,
                        model=model,
                        tile=tile,
                        executable=realesrgan,
                        gpu_id=gpu_id,
                        callback=callback,
                    )
                    current = upscaled
                except Exception as exc:
                    _status(callback, f"Real-ESRGAN skipped ({exc}); using high-quality Lanczos delivery scaling")
            else:
                _status(callback, "Real-ESRGAN not installed; using high-quality Lanczos delivery scaling")

        _status(callback, f"Encoding final {width}x{height} delivery at {actual_fps} FPS")
        return export_delivery(
            current,
            output,
            int(width),
            int(height),
            enhance_quality=True,
            target_fps=actual_fps,
        )
