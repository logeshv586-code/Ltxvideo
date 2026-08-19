import unittest

from engine.skill_engine import SKILL_ENGINE, VideoRequest


class SkillEngineTests(unittest.TestCase):
    def test_preserves_user_prompt_exactly(self):
        raw = 'A red robot says "GO" and runs toward camera.'
        plan = SKILL_ENGINE.plan(VideoRequest(raw_prompt=raw))
        self.assertTrue(plan.prompt.startswith(raw))
        self.assertIn("GO", plan.locked_terms)
        self.assertIn("red", [item.lower() for item in plan.locked_terms])

    def test_reference_skill_is_mandatory_for_i2v(self):
        plan = SKILL_ENGINE.plan(VideoRequest(raw_prompt="She smiles.", has_reference=True))
        self.assertIn("reference-anchor", plan.applied_skills)
        self.assertIn("authoritative", plan.prompt)

    def test_action_short_clip_limits_complexity(self):
        plan = SKILL_ENGINE.plan(VideoRequest(
            raw_prompt="Action environment: rooftop. Energy: Dynamic. A runner jumps a gap.",
            mode="auto",
            duration_seconds=4.0,
        ))
        self.assertIn("action-consistency", plan.applied_skills)
        self.assertIn("one major camera movement", plan.prompt)

    def test_user_camera_instruction_is_locked(self):
        plan = SKILL_ENGINE.plan(VideoRequest(raw_prompt="Slow dolly in as the actor turns."))
        self.assertIn("camera-lock", plan.applied_skills)
        self.assertNotIn("unrequested camera move", plan.prompt)

    def test_negative_prompt_is_hardened_without_duplicates(self):
        plan = SKILL_ENGINE.plan(VideoRequest(
            raw_prompt="A woman walks.",
            mode="real_world",
            negative_prompt="warped anatomy, blurry",
        ))
        self.assertEqual(plan.negative_prompt.lower().count("warped anatomy"), 1)
        self.assertIn("impossible physics", plan.negative_prompt.lower())

    def test_compiled_prompt_stays_in_ltx_guidance_range(self):
        plan = SKILL_ENGINE.plan(VideoRequest(
            raw_prompt="A fox runs through snow.",
            mode="cartoon",
            has_reference=True,
        ))
        self.assertLessEqual(len(plan.prompt.split()), 200)


if __name__ == "__main__":
    unittest.main()
