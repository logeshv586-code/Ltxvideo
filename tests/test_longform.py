import unittest

from engine.longform import (
    ASPECT_LABELS,
    CLIP_LENGTHS,
    CUSTOMER_QUALITY_CHOICES,
    QUALITY_PROFILES,
    estimate_auto_seconds,
    extract_prompt_directives,
    plan_story,
    scene_prompt,
)


class PersonalVideoPlannerTests(unittest.TestCase):
    def test_auto_duration_scaling_helper_still_caps_at_ten_minutes(self):
        self.assertGreaterEqual(estimate_auto_seconds("A fox opens a door."), 8)
        self.assertLessEqual(estimate_auto_seconds("word " * 2000), 600)

    def test_single_four_second_clip(self):
        plan = plan_story(
            "A fox walks across the moon.",
            "15 seconds",
            "High",
            "Landscape • 1280×720 • 16:9",
            generation_mode="Single Clip",
            clip_length_label="4 seconds • Recommended",
        )
        self.assertEqual(plan.scene_count, 1)
        self.assertEqual(plan.clip_frames, 97)
        self.assertEqual(plan.target_seconds, 4)
        self.assertEqual(plan.generation_mode, "Single Clip")
        self.assertEqual((plan.width, plan.height), (576, 320))
        self.assertEqual(plan.profile.inference_steps, 28)

    def test_single_eight_second_high_uses_memory_safe_native_size(self):
        plan = plan_story(
            "A fox walks across the moon.",
            "15 seconds",
            "High",
            "Landscape • 1280×720 • 16:9",
            generation_mode="Single Clip",
            clip_length_label="8 seconds • Max practical",
        )
        self.assertEqual(plan.scene_count, 1)
        self.assertEqual(plan.clip_frames, 193)
        self.assertEqual(plan.target_seconds, 8)
        self.assertEqual((plan.width, plan.height), (512, 288))
        self.assertIn("memory-safe", plan.profile.label)

    def test_continuous_fifteen_seconds_uses_gpu_sized_chunks(self):
        story = "On the moon a fox meets a cow. Then they make a cookie. After that they walk together."
        plan = plan_story(
            story,
            "15 seconds",
            "Balanced",
            "Landscape • 1280×720 • 16:9",
            generation_mode="Continuous Video",
            clip_length_label="4 seconds • Recommended",
        )
        self.assertEqual(plan.target_seconds, 15)
        self.assertEqual(plan.scene_count, 4)
        self.assertEqual(plan.continuity_mode, "continuous")
        self.assertTrue(any("cookie" in beat.lower() for beat in plan.beats))

    def test_five_minutes_stays_within_scene_cap(self):
        plan = plan_story(
            "A traveler crosses changing landscapes and meets different people along the journey.",
            "5 minutes",
            "Balanced",
            "Landscape • 1280×720 • 16:9",
            generation_mode="Continuous Video",
            clip_length_label="4 seconds • Recommended",
        )
        self.assertLessEqual(plan.scene_count, 96)
        self.assertGreaterEqual(plan.estimated_seconds, 295)

    def test_customer_ui_hides_fast_draft_profile(self):
        self.assertNotIn("Fast", CUSTOMER_QUALITY_CHOICES)
        self.assertEqual(CUSTOMER_QUALITY_CHOICES, ("Balanced", "High"))
        self.assertIn("Draft only", QUALITY_PROFILES["Fast"].label)

    def test_all_native_sizes_and_clip_lengths_are_ltx_aligned(self):
        for profile in QUALITY_PROFILES.values():
            for size in (profile.landscape, profile.portrait, profile.square):
                self.assertEqual(size[0] % 32, 0)
                self.assertEqual(size[1] % 32, 0)
            self.assertEqual(profile.fps, 24)
            self.assertEqual((profile.tail_frames - 1) % 8, 0)
        for frames in CLIP_LENGTHS.values():
            self.assertEqual((frames - 1) % 8, 0)
            self.assertLessEqual(frames + 16, 300)

    def test_customer_aspects_resolve(self):
        for label, expected in ASPECT_LABELS.items():
            plan = plan_story(
                "A product rotates on a studio table.",
                "15 seconds",
                "Balanced",
                label,
            )
            self.assertEqual(plan.aspect, expected)

    def test_prompt_directives_are_removed_from_visible_story(self):
        text = (
            "A moon cookie sits on a cloud and smiles.\n"
            "Style: premium handcrafted clay animation\n"
            "Duration: 8-10 seconds\n"
            "Camera: slow zoom in"
        )
        cleaned, directives = extract_prompt_directives(text)
        self.assertNotIn("Duration", cleaned)
        self.assertNotIn("Style:", cleaned)
        self.assertIn("clay animation", directives["style"])
        self.assertEqual(directives["camera"], "slow zoom in")

    def test_single_clip_does_not_receive_entire_long_script(self):
        text = (
            "The Moon Cookie sits on a cloud and speaks. Then it jumps to another cloud. "
            "Then a fox enters. Then they make a cookie. Then everybody walks away together.\n"
            "Style: clay animation"
        )
        plan = plan_story(
            text,
            "15 seconds",
            "High",
            "Landscape • 1280×720 • 16:9",
            generation_mode="Single Clip",
            clip_length_label="4 seconds • Recommended",
        )
        self.assertEqual(plan.scene_count, 1)
        self.assertLessEqual(len(plan.beats[0].split()), 44)
        self.assertIn("Moon Cookie", plan.beats[0])
        self.assertNotIn("Style:", plan.story)
        self.assertEqual(plan.style_hint, "clay animation")

    def test_continuation_prompt_keeps_concise_story_and_identity(self):
        prompt = scene_prompt(
            "The fox helps make a cookie",
            1,
            4,
            "premium 3D animation",
            "The fox is orange with green eyes",
            continuity_mode="continuous",
            full_story="A fox meets a cow on the moon and they make a cookie together.",
            camera_hint="slow tracking shot",
        )
        self.assertIn("Continue directly", prompt)
        self.assertIn("orange", prompt)
        self.assertNotIn("Concise overall context", prompt)
        self.assertIn("cookie", prompt)
        self.assertIn("slow tracking", prompt)
        self.assertLessEqual(len(prompt.split()), 90)


if __name__ == "__main__":
    unittest.main()
