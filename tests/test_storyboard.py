import unittest
from engine.storyboard import build_scene_prompt, split_story_beats


class StoryboardTests(unittest.TestCase):
    def test_scene_count_is_exact(self):
        beats = split_story_beats("One. Two.", 4)
        self.assertEqual(len(beats), 4)

    def test_continuation_prompt_locks_character(self):
        prompt = build_scene_prompt("Milo opens the door", 1, 4, "Claymation", "Milo: orange fox")
        self.assertIn("preserve identical character", prompt)
        self.assertIn("Milo: orange fox", prompt)


if __name__ == "__main__":
    unittest.main()
