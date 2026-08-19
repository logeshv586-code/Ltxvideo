import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from engine.video_qc import inspect_video


class VideoQCTests(unittest.TestCase):
    def _make_video(self, changing: bool):
        temp_dir = tempfile.TemporaryDirectory()
        path = Path(temp_dir.name) / "sample.mp4"
        writer = cv2.VideoWriter(
            str(path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            10,
            (64, 64),
        )
        for index in range(20):
            value = index * 8 if changing else 40
            writer.write(np.full((64, 64, 3), value, dtype=np.uint8))
        writer.release()
        return temp_dir, path

    def test_valid_video_decodes(self):
        temp_dir, path = self._make_video(changing=True)
        try:
            report = inspect_video(path, 64, 64, 2.0)
            self.assertFalse(report.fatal)
            self.assertEqual(report.frame_count, 20)
        finally:
            temp_dir.cleanup()

    def test_static_video_is_flagged(self):
        temp_dir, path = self._make_video(changing=False)
        try:
            report = inspect_video(path)
            self.assertGreaterEqual(report.static_ratio, 0.85)
            self.assertTrue(any("frozen" in warning for warning in report.warnings))
        finally:
            temp_dir.cleanup()


if __name__ == "__main__":
    unittest.main()
