"""Low-memory video utilities for LTX Cartoon Studio."""
from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import cv2
from PIL import Image


def extract_last_frame(video_path: str | Path) -> Image.Image:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, total - 1))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise ValueError(f"Cannot read last frame from: {video_path}")
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


def get_video_info(video_path: str | Path) -> dict:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    info = {
        "fps": fps,
        "frame_count": frames,
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "duration": frames / fps if fps else 0,
    }
    cap.release()
    return info


def _ffmpeg_exe() -> str:
    system = shutil.which("ffmpeg")
    if system:
        return system
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:
        raise RuntimeError("FFmpeg is required. Install ffmpeg or imageio-ffmpeg.") from exc


def concatenate_videos_streaming(
    video_paths: list[str | Path], output_path: str | Path, target_fps: int = 30
) -> Path:
    """Concatenate clips without loading every frame into 16 GB system RAM."""
    paths = [Path(p).resolve() for p in video_paths]
    if not paths:
        raise ValueError("No videos to concatenate")
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if len(paths) == 1:
        shutil.copy2(paths[0], output_path)
        return output_path

    with tempfile.TemporaryDirectory() as td:
        concat_file = Path(td) / "clips.txt"
        concat_file.write_text(
            "\n".join(f"file '{str(p)}'" for p in paths),
            encoding="utf-8",
        )
        cmd = [
            _ffmpeg_exe(), "-y", "-f", "concat", "-safe", "0", "-i", str(concat_file),
            "-r", str(target_fps), "-c:v", "libx264", "-preset", "medium", "-crf", "18",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output_path),
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    return output_path


def export_delivery(video_path: str | Path, output_path: str | Path, width: int, height: int) -> Path:
    """Create a delivery-size MP4. This is a high-quality resize, not AI detail synthesis."""
    video_path, output_path = Path(video_path), Path(output_path)
    cmd = [
        _ffmpeg_exe(), "-y", "-i", str(video_path),
        "-vf", f"scale={width}:{height}:force_original_aspect_ratio=decrease,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2",
        "-c:v", "libx264", "-preset", "slow", "-crf", "17", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", str(output_path),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    return output_path
