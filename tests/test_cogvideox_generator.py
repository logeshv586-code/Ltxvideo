import unittest

from engine.cogvideox_generator import (
    COGVIDEOX_FPS,
    COGVIDEOX_FRAMES,
    COGVIDEOX_OFFLOAD_MODE,
    CogVideoXWorker,
    build_story_sequence,
)


class CogVideoXGeneratorTests(unittest.TestCase):
    def test_official_clip_profile_is_six_seconds_at_eight_fps(self):
        self.assertEqual(COGVIDEOX_FPS, 8)
        self.assertEqual(COGVIDEOX_FRAMES, 49)
        self.assertAlmostEqual(COGVIDEOX_FRAMES / COGVIDEOX_FPS, 6.125)

    def test_t4_safe_sequential_offload_is_the_default(self):
        self.assertEqual(COGVIDEOX_OFFLOAD_MODE, "sequential")

    def test_negative_seed_is_randomized(self):
        seed = CogVideoXWorker._seed(-1)
        self.assertGreaterEqual(seed, 0)
        self.assertLess(seed, 2**31)

    def test_explicit_seed_is_retained(self):
        self.assertEqual(CogVideoXWorker._seed(42), 42)

    def test_story_sequence_preserves_explicit_action_order(self):
        beats = build_story_sequence(
            "The Fox finds a cookie. Then the Fox opens the cookie. Finally the Fox shares the cookie.",
            15,
        )
        self.assertEqual(len(beats), 3)
        self.assertIn("finds", beats[0])
        self.assertIn("opens", beats[1])
        self.assertIn("shares", beats[2])

    def test_story_sequence_extends_a_short_prompt_to_fill_duration(self):
        beats = build_story_sequence("A Fox walks through the garden.", 15)
        self.assertEqual(len(beats), 3)
        self.assertIn("Continue naturally", beats[1])

    def test_story_sequence_respects_the_selected_model_clip_length(self):
        beats = build_story_sequence("A Fox walks through the garden.", 15, clip_seconds=5.0625)
        self.assertEqual(len(beats), 3)


if __name__ == "__main__":
    unittest.main()
