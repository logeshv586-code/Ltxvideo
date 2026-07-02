"""
LTX-2.3 Video Generation Platform — Video Processor
Frame extraction, video concatenation, crossfade transitions, and encoding.
"""

import tempfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def extract_last_frame(video_path: str | Path) -> Image.Image:
    """Extract the last frame from a video file as a PIL Image."""
    video_path = str(video_path)
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    # Seek to last frame
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames - 1)

    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise ValueError(f"Cannot read last frame from: {video_path}")

    # Convert BGR to RGB and to PIL
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(frame_rgb)


def extract_frame_at(video_path: str | Path, frame_index: int) -> Image.Image:
    """Extract a specific frame from a video file."""
    video_path = str(video_path)
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise ValueError(f"Cannot read frame {frame_index} from: {video_path}")

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(frame_rgb)


def get_video_info(video_path: str | Path) -> dict:
    """Get video metadata (fps, frame count, duration, resolution)."""
    video_path = str(video_path)
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    info = {
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    info["duration"] = info["frame_count"] / info["fps"] if info["fps"] > 0 else 0
    cap.release()
    return info


def concatenate_videos(
    video_paths: list[str | Path],
    output_path: str | Path,
    crossfade_frames: int = 12,
    target_fps: float = 24.0,
) -> Path:
    """
    Concatenate multiple video clips with crossfade transitions.

    Args:
        video_paths: List of video file paths to concatenate
        output_path: Output file path
        crossfade_frames: Number of frames for crossfade (0 = hard cut)
        target_fps: Output FPS
    """
    output_path = Path(output_path)

    if len(video_paths) == 0:
        raise ValueError("No videos to concatenate")

    if len(video_paths) == 1:
        # Single video, just copy
        import shutil
        shutil.copy2(str(video_paths[0]), str(output_path))
        return output_path

    # Read all video frames
    all_clips_frames = []
    for vpath in video_paths:
        frames = _read_video_frames(str(vpath))
        if len(frames) > 0:
            all_clips_frames.append(frames)

    if len(all_clips_frames) == 0:
        raise ValueError("No valid frames found in any video")

    # Apply crossfade and concatenate
    merged_frames = all_clips_frames[0]

    for i in range(1, len(all_clips_frames)):
        next_clip = all_clips_frames[i]
        cf = min(crossfade_frames, len(merged_frames), len(next_clip))

        if cf > 0:
            # Create crossfade transition
            for j in range(cf):
                alpha = j / cf
                blended = cv2.addWeighted(
                    merged_frames[-(cf - j)], 1 - alpha,
                    next_clip[j], alpha,
                    0,
                )
                merged_frames[-(cf - j)] = blended

            # Append remaining frames (skip the crossfade overlap)
            merged_frames.extend(next_clip[cf:])
        else:
            merged_frames.extend(next_clip)

    # Write output video
    _write_video_frames(merged_frames, str(output_path), target_fps)
    return output_path


def _read_video_frames(video_path: str) -> list[np.ndarray]:
    """Read all frames from a video file."""
    cap = cv2.VideoCapture(video_path)
    frames = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    cap.release()
    return frames


def _write_video_frames(
    frames: list[np.ndarray],
    output_path: str,
    fps: float = 24.0,
) -> None:
    """Write frames to a video file using H.264 codec."""
    if len(frames) == 0:
        raise ValueError("No frames to write")

    h, w = frames[0].shape[:2]

    # Try H.264 codec, fallback to mp4v
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    if not writer.isOpened():
        raise ValueError(f"Cannot create video writer for: {output_path}")

    for frame in frames:
        writer.write(frame)

    writer.release()


def create_thumbnail(video_path: str | Path, size: tuple[int, int] = (320, 180)) -> Image.Image:
    """Create a thumbnail from the first frame of a video."""
    frame = extract_frame_at(video_path, 0)
    frame.thumbnail(size, Image.Resampling.LANCZOS)
    return frame


def frames_to_video(
    frames: list[np.ndarray] | np.ndarray,
    output_path: str | Path,
    fps: float = 24.0,
) -> Path:
    """Save a list of numpy frames (RGB) as an MP4 video."""
    output_path = Path(output_path)

    if isinstance(frames, np.ndarray):
        # Shape: (T, H, W, C) or (T, C, H, W)
        if frames.ndim == 4:
            if frames.shape[1] == 3:  # (T, C, H, W)
                frames = np.transpose(frames, (0, 2, 3, 1))
            frame_list = [cv2.cvtColor(f, cv2.COLOR_RGB2BGR) for f in frames]
        else:
            raise ValueError(f"Unexpected frame array shape: {frames.shape}")
    else:
        frame_list = []
        for f in frames:
            if isinstance(f, Image.Image):
                f = np.array(f)
            if f.ndim == 3 and f.shape[2] == 3:
                f = cv2.cvtColor(f, cv2.COLOR_RGB2BGR)
            frame_list.append(f)

    _write_video_frames(frame_list, str(output_path), fps)
    return output_path
