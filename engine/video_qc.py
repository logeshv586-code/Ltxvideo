"""Lightweight technical QC for generated MP4 files.

This does not pretend to judge semantic prompt accuracy. It checks technical
failures we can verify locally: decodability, duration, dimensions, black frames,
and near-frozen output.
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
    duration: float = 0.0
    fps: float = 0.0
    frame_count: int = 0
    width: int = 0
    height: int = 0
    static_ratio: float = 0.0
    black_ratio: float = 0.0
    warnings: list[str] = field(default_factory=list)

    def summary(self) -> str:
        state = "PASS" if self.passed else ("FAIL" if self.fatal else "WARN")
        lines = [
            f"Video QC: {state}",
            f"{self.width}×{self.height} · {self.duration:.2f}s · {self.fps:.1f} fps · {self.frame_count} frames",
        ]
        if self.static_ratio:
            lines.append(f"Near-static sample ratio: {self.static_ratio:.0%}")
        if self.black_ratio:
            lines.append(f"Dark/black sample ratio: {self.black_ratio:.0%}")
        lines.extend(f"QC warning: {w}" for w in self.warnings)
        return "\n".join(lines)


def inspect_video(
    video_path: str | Path,
    expected_width: int | None = None,
    expected_height: int | None = None,
    expected_duration: float | None = None,
    sample_count: int = 10,
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
    if frame_count > 1:
        indices = np.linspace(0, max(0, frame_count - 1), num=min(sample_count, frame_count), dtype=int)
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame = cap.read()
            if ok and frame is not None:
                small = cv2.resize(frame, (160, 90), interpolation=cv2.INTER_AREA)
                samples.append(small)
    cap.release()

    black_ratio = 0.0
    static_ratio = 0.0
    if samples:
        black_ratio = sum(float(np.mean(frame)) < 5.0 for frame in samples) / len(samples)
        diffs = [float(np.mean(cv2.absdiff(a, b))) for a, b in zip(samples, samples[1:])]
        if diffs:
            static_ratio = sum(diff < 0.9 for diff in diffs) / len(diffs)
        if black_ratio >= 0.8:
            warnings.append("Most sampled frames are effectively black.")
            fatal = True
        elif black_ratio >= 0.4:
            warnings.append("Many sampled frames are very dark; review the render.")
        if static_ratio >= 0.85:
            warnings.append("The video appears nearly frozen across sampled frames.")
        elif static_ratio >= 0.6:
            warnings.append("Motion is low across many sampled frames; review whether the requested action occurred.")

    passed = not fatal and not warnings
    return VideoQCReport(
        passed=passed,
        fatal=fatal,
        duration=duration,
        fps=fps,
        frame_count=frame_count,
        width=width,
        height=height,
        static_ratio=static_ratio,
        black_ratio=black_ratio,
        warnings=warnings,
    )
