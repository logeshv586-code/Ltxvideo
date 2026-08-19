import unittest

from engine.prompt_builders import (
    MAX_PROMPT_WORDS,
    build_action_prompt,
    build_comics_prompt,
    build_directed_prompt,
    build_real_world_prompt,
)


class PromptBuilderTests(unittest.TestCase):
    def test_comics_contains_style_and_identity_lock(self):
        prompt = build_comics_prompt(
            "A masked detective", "steps from the shadows", "a rainy neon alley",
            "Graphic Novel", "Medium shot", "Gentle dolly in", "Neon night lighting",
        )
        self.assertIn("graphic novel", prompt.lower())
        self.assertIn("consistent line work", prompt.lower())

    def test_real_world_enforces_physical_realism(self):
        prompt = build_real_world_prompt(
            "A chef", "places a plated dish on the counter", "a working restaurant kitchen at night",
            "Cinematic Live Action", "Medium shot", "Tripod-locked camera", "Warm practical interior lighting",
        )
        self.assertIn("believable weight and inertia", prompt.lower())
        self.assertIn("realistic materials", prompt.lower())

    def test_action_short_clip_limits_complexity(self):
        prompt = build_action_prompt(
            "A parkour athlete", "vaults over a concrete barrier", "an empty rooftop",
            "Parkour", "Tracking side profile", "Stable lateral tracking", "Golden-hour backlight",
            duration_seconds=4.0,
        )
        self.assertIn("one primary action beat", prompt.lower())
        self.assertIn("do not add a second unrelated stunt", prompt.lower())

    def test_reference_mode_mentions_visual_anchor(self):
        prompt = build_directed_prompt(
            "real_world", "A cyclist", "begins pedaling", "a country road", "Documentary",
            "Medium shot", "Stable lateral tracking", "Soft natural daylight", has_reference=True,
        )
        self.assertIn("supplied reference image", prompt.lower())
        self.assertIn("describe only the new motion", prompt.lower())

    def test_prompt_word_limit(self):
        verbose = "detail " * 400
        prompt = build_comics_prompt(
            verbose, verbose, verbose, "Motion Comic", "Wide establishing shot",
            "Slow cinematic pan", "Soft natural daylight", verbose, verbose,
        )
        self.assertLessEqual(len(prompt.split()), MAX_PROMPT_WORDS)

    def test_unknown_mode_fails(self):
        with self.assertRaises(ValueError):
            build_directed_prompt("unknown", "x", "y", "z", "s", "shot", "cam", "light")


if __name__ == "__main__":
    unittest.main()
