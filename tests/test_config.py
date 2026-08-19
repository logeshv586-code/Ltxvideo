import unittest
from config import DURATION_PRESETS, MAX_NATIVE_FRAMES, RESOLUTION_PRESETS


class ConfigTests(unittest.TestCase):
    def test_resolution_alignment(self):
        for preset in RESOLUTION_PRESETS.values():
            self.assertEqual(preset["width"] % 32, 0)
            self.assertEqual(preset["height"] % 32, 0)

    def test_frame_alignment_and_cap(self):
        for frames in DURATION_PRESETS.values():
            self.assertEqual((frames - 1) % 8, 0)
            self.assertLessEqual(frames, MAX_NATIVE_FRAMES)


if __name__ == "__main__":
    unittest.main()
