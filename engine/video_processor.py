"""Low-memory video utilities for LTX Personal Video Maker."""
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


def extract_tail_frames(
    video_path: str | Path,
    frame_count: int = 17,
    safety_margin: int = 1,
) -> list[Image.Image]:
    frame_count = max(1, int(frame_count))
    safety_margin = max(0, int(safety_margin))
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total <= 0:
        cap.release()
        raise ValueError(f"Video has no readable frames: {video_path}")

    end_exclusive = max(1, total - safety_margin)
    start = max(0, end_exclusive - frame_count)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    frames: list[Image.Image] = []
    for _ in range(start, end_exclusive):
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    cap.release()
    if not frames:
        raise ValueError(f"Cannot extract continuation frames from: {video_path}")
    return frames


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


def trim_video_start_frames(
    video_path: str | Path,
    output_path: str | Path,
    frames_to_trim: int,
    target_fps: int = 24,
) -> Path:
    video_path = Path(video_path).resolve()
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frames_to_trim = max(0, int(frames_to_trim))
    if frames_to_trim == 0:
        shutil.copy2(video_path, output_path)
        return output_path

    vf = f"trim=start_frame={frames_to_trim},setpts=PTS-STARTPTS,fps={int(target_fps)}"
    cmd = [
        _ffmpeg_exe(), "-y", "-i", str(video_path),
        "-vf", vf, "-an",
        "-c:v", "libx264", "-preset", "slow", "-crf", "14",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output_path),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    return output_path


def concatenate_videos_streaming(
    video_paths: list[str | Path], output_path: str | Path, target_fps: int = 24
) -> Path:
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
        concat_file.write_text("\n".join(f"file '{str(p)}'" for p in paths), encoding="utf-8")
        cmd = [
            _ffmpeg_exe(), "-y", "-f", "concat", "-safe", "0", "-i", str(concat_file),
            "-r", str(int(target_fps)),
            "-c:v", "libx264", "-preset", "slow",
            "-b:v", "4M", "-maxrate", "6M", "-bufsize", "12M",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output_path),
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    return output_path


def export_delivery(
    video_path: str | Path,
    output_path: str | Path,
    width: int,
    height: int,
    enhance_quality: bool = True,
    target_fps: int = 24,
    duration_seconds: float | None = None,
) -> Path:
    """Create the customer delivery MP4 and optionally trim to exact duration.

    The previous delivery target could still throw away fine texture after an
    expensive generation. This profile gives 720p final renders more bitrate
    headroom without pretending post-processing can invent model detail.
    """
    video_path, output_path = Path(video_path), Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    scale_filter = (
        f"scale={width}:{height}:flags=lanczos:force_original_aspect_ratio=decrease,"
        f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,fps={int(target_fps)}"
    )
    # Keep sharpening light. Strong sharpening cannot recreate detail that was
    # not generated and creates halos around stylized animation silhouettes.
    vf_pipeline = f"{scale_filter},unsharp=5:5:0.22:5:5:0.0" if enhance_quality else scale_filter

    cmd = [_ffmpeg_exe(), "-y", "-i", str(video_path)]
    if duration_seconds is not None and float(duration_seconds) > 0:
        cmd.extend(["-t", f"{float(duration_seconds):.3f}"])
    cmd.extend([
        "-vf", vf_pipeline,
        "-c:v", "libx264", "-preset", "slow",
        "-crf", "17", "-maxrate", "10M", "-bufsize", "20M",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output_path),
    ])
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    return output_path
