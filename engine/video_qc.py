"""Lightweight technical and visual QC for generated MP4 files.

The QC gate does not claim semantic understanding. It detects measurable
failure modes that were visible in long-form recursive generations: black or
frozen output, severe sharpness collapse near the tail, and extremely weak
motion. Long-form generation can retry a scene before a damaged tail is fed
back into the next continuation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np


@dataclass
class VideoQCReport:
    passed: bool
    fatal: bool
    visual_failure: bool = False
    duration: float = 0.0
    fps: float = 0.0
    frame_count: int = 0
    width: int = 0
    height: int = 0
    static_ratio: float = 0.0
    black_ratio: float = 0.0
    mean_sharpness: float = 0.0
    tail_sharpness: float = 0.0
    warnings: list[str] = field(default_factory=list)

    def summary(self) -> str:
        state = "PASS" if self.passed else ("FAIL" if self.fatal or self.visual_failure else "WARN")
        lines = [
            f"Video QC: {state}",
            f"{self.width}×{self.height} · {self.duration:.2f}s · {self.fps:.1f} fps · {self.frame_count} frames",
        ]
        if self.mean_sharpness:
            lines.append(
                f"Visual sharpness: mean {self.mean_sharpness:.1f} · tail {self.tail_sharpness:.1f}"
            )
        if self.static_ratio:
            lines.append(f"Near-static sample ratio: {self.static_ratio:.0%}")
        if self.black_ratio:
            lines.append(f"Dark/black sample ratio: {self.black_ratio:.0%}")
        lines.extend(f"QC warning: {w}" for w in self.warnings)
        return "\n".join(lines)


def _sharpness(frame: np.ndarray) -> float:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def inspect_video(
    video_path: str | Path,
    expected_width: int | None = None,
    expected_height: int | None = None,
    expected_duration: float | None = None,
    sample_count: int = 12,
) -> VideoQCReport:
    path = Path(video_path)
    if not path.exists() or path.stat().st_size == 0:
        return VideoQCReport(False, True, warnings=["Output file is missing or empty."])

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return VideoQCReport(False, True, warnings=["Output file cannot be decoded."])

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    duration = frame_count / fps if fps > 0 else 0.0
    warnings: list[str] = []
    fatal = frame_count <= 1 or fps <= 0 or width <= 0 or height <= 0
    visual_failure = False

    if expected_width and expected_height and (width != expected_width or height != expected_height):
        warnings.append(
            f"Native output size {width}×{height} differs from requested {expected_width}×{expected_height}."
        )
    if expected_duration and duration < expected_duration * 0.60:
        warnings.append(
            f"Output duration {duration:.2f}s is much shorter than the requested ~{expected_duration:.2f}s."
        )
        fatal = True

    samples: list[np.ndarray] = []
    sample_indices: list[int] = []
    if frame_count > 1:
        sample_indices = np.linspace(
            0,
            max(0, frame_count - 1),
            num=min(sample_count, frame_count),
            dtype=int,
        ).tolist()
        for idx in sample_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame = cap.read()
            if ok and frame is not None:
                small = cv2.resize(frame, (192, 108), interpolation=cv2.INTER_AREA)
                samples.append(small)

    # Inspect a contiguous tail separately because that is what the long-form
    # continuation path will condition on.
    tail_samples: list[np.ndarray] = []
    if frame_count > 2:
        tail_count = min(6, frame_count)
        start = max(0, frame_count - tail_count - 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        for _ in range(tail_count):
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            tail_samples.append(cv2.resize(frame, (192, 108), interpolation=cv2.INTER_AREA))
    cap.release()

    black_ratio = 0.0
    static_ratio = 0.0
    mean_sharpness = 0.0
    tail_sharpness = 0.0

    if samples:
        black_ratio = sum(float(np.mean(frame)) < 5.0 for frame in samples) / len(samples)
        diffs = [float(np.mean(cv2.absdiff(a, b))) for a, b in zip(samples, samples[1:])]
        if diffs:
            static_ratio = sum(diff < 0.9 for diff in diffs) / len(diffs)

        sharpness_values = [_sharpness(frame) for frame in samples]
        mean_sharpness = float(np.mean(sharpness_values)) if sharpness_values else 0.0
        tail_values = [_sharpness(frame) for frame in tail_samples]
        tail_sharpness = float(np.mean(tail_values)) if tail_values else mean_sharpness

        if black_ratio >= 0.8:
            warnings.append("Most sampled frames are effectively black.")
            fatal = True
        elif black_ratio >= 0.4:
            warnings.append("Many sampled frames are very dark; review the render.")

        if static_ratio >= 0.90:
            warnings.append("The video is effectively frozen across sampled frames.")
            visual_failure = True
        elif static_ratio >= 0.65:
            warnings.append("Motion is low across many sampled frames; review whether the requested action occurred.")

        # Use a relative test rather than a universal absolute blur threshold,
        # because stylized animation naturally has lower Laplacian variance than
        # photographic footage. The tail must collapse dramatically before this
        # gate triggers an automatic retry.
        if mean_sharpness >= 10.0 and tail_sharpness < mean_sharpness * 0.25:
            warnings.append("Severe sharpness collapse detected near the scene tail.")
            visual_failure = True
        elif mean_sharpness >= 10.0 and tail_sharpness < mean_sharpness * 0.45:
            warnings.append("The scene becomes noticeably softer near the tail.")

        if tail_sharpness < 3.0 and mean_sharpness >= 8.0:
            warnings.append("Continuation tail is too blurred to reuse safely.")
            visual_failure = True

    passed = not fatal and not visual_failure and not warnings
    return VideoQCReport(
        passed=passed,
        fatal=fatal,
        visual_failure=visual_failure,
        duration=duration,
        fps=fps,
        frame_count=frame_count,
        width=width,
        height=height,
        static_ratio=static_ratio,
        black_ratio=black_ratio,
        mean_sharpness=mean_sharpness,
        tail_sharpness=tail_sharpness,
        warnings=warnings,
    )
