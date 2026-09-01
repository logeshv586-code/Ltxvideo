import unittest

from engine.longform import (
    ASPECT_LABELS,
    QUALITY_PROFILES,
    estimate_auto_seconds,
    plan_story,
    scene_prompt,
)


class LongFormPlannerTests(unittest.TestCase):
    def test_auto_duration_scales_with_story_length(self):
        short = estimate_auto_seconds("A fox opens a door and smiles.")
        long = estimate_auto_seconds("word " * 700)
        self.assertGreaterEqual(short, 8)
        self.assertGreater(long, short)
        self.assertLessEqual(long, 300)

    def test_five_minute_fast_plan_fits_scene_cap(self):
        story = "A traveler crosses a changing landscape while the camera follows the journey. " * 20
        plan = plan_story(
            story,
            "5 minutes",
            "Fast",
            "YouTube / Landscape (16:9)",
        )
        self.assertLessEqual(plan.scene_count, 96)
        self.assertGreaterEqual(plan.estimated_seconds, 295)
        self.assertEqual(plan.aspect, "16:9")

    def test_all_native_sizes_are_ltx_aligned(self):
        for profile in QUALITY_PROFILES.values():
            for size in (profile.landscape, profile.portrait, profile.square):
                self.assertEqual(size[0] % 32, 0)
                self.assertEqual(size[1] % 32, 0)
                self.assertEqual((profile.frames_per_scene - 1) % 8, 0)

    def test_all_customer_aspects_resolve(self):
        story = "A product rotates slowly on a studio table while light travels across its surface."
        for label, expected in ASPECT_LABELS.items():
            plan = plan_story(story, "15 seconds", "Balanced", label)
            self.assertEqual(plan.aspect, expected)

    def test_scene_prompt_has_explicit_continuity(self):
        prompt = scene_prompt(
            "Milo reaches for the glowing key",
            1,
            5,
            "premium 3D animation",
            "Milo is an orange fox with a teal scarf",
        )
        self.assertIn("Direct continuation", prompt)
        self.assertIn("orange fox", prompt)
        self.assertIn("end on a stable readable pose", prompt)


if __name__ == "__main__":
    unittest.main()
