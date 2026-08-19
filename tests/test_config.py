import unittest
from config import (
    ACTION_STYLES,
    COMICS_STYLES,
    DURATION_PRESETS,
    MAX_NATIVE_FRAMES,
    REAL_WORLD_STYLES,
    RESOLUTION_PRESETS,
)


class ConfigTests(unittest.TestCase):
    def test_resolution_alignment(self):
        for preset in RESOLUTION_PRESETS.values():
            self.assertEqual(preset["width"] % 32, 0)
            self.assertEqual(preset["height"] % 32, 0)

    def test_frame_alignment_and_cap(self):
        for frames in DURATION_PRESETS.values():
            self.assertEqual((frames - 1) % 8, 0)
            self.assertLessEqual(frames, MAX_NATIVE_FRAMES)

    def test_directed_studio_presets_exist(self):
        self.assertGreaterEqual(len(COMICS_STYLES), 4)
        self.assertGreaterEqual(len(REAL_WORLD_STYLES), 4)
        self.assertGreaterEqual(len(ACTION_STYLES), 4)


if __name__ == "__main__":
    unittest.main()
