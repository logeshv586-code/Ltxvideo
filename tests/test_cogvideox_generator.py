import unittest

from engine.cogvideox_generator import COGVIDEOX_FPS, COGVIDEOX_FRAMES, CogVideoXWorker


class CogVideoXGeneratorTests(unittest.TestCase):
    def test_official_clip_profile_is_six_seconds_at_eight_fps(self):
        self.assertEqual(COGVIDEOX_FPS, 8)
        self.assertEqual(COGVIDEOX_FRAMES, 49)
        self.assertAlmostEqual(COGVIDEOX_FRAMES / COGVIDEOX_FPS, 6.125)

    def test_negative_seed_is_randomized(self):
        seed = CogVideoXWorker._seed(-1)
        self.assertGreaterEqual(seed, 0)
        self.assertLess(seed, 2**31)

    def test_explicit_seed_is_retained(self):
        self.assertEqual(CogVideoXWorker._seed(42), 42)


if __name__ == "__main__":
    unittest.main()
