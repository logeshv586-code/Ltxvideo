import unittest

from engine.audio_processor import (
    auto_narration_from_prompt,
    detect_voice_language,
    resolve_narration,
)


class AudioProcessorTests(unittest.TestCase):
    def test_auto_detects_tamil_script(self):
        self.assertEqual(detect_voice_language("வணக்கம்! இது ஒரு கதை."), "Tamil")

    def test_auto_detects_english(self):
        self.assertEqual(detect_voice_language("A fox walks through a forest."), "English")

    def test_explicit_narration_overrides_visual_prompt(self):
        prompt = "A fox walks through a forest.\nStyle: cinematic\nNarration: Welcome to the forest."
        self.assertEqual(auto_narration_from_prompt(prompt), "Welcome to the forest.")

    def test_auto_narration_omits_generation_directives(self):
        prompt = "A girl opens a glowing book in a quiet library.\nCamera: slow dolly in"
        narration, language = resolve_narration(prompt)
        self.assertEqual(narration, "A girl opens a glowing book in a quiet library.")
        self.assertEqual(language, "English")

    def test_no_voice_mode_has_no_narration(self):
        narration, language = resolve_narration("வணக்கம் உலகம்", mode="No voice")
        self.assertEqual(narration, "")
        self.assertEqual(language, "Tamil")


if __name__ == "__main__":
    unittest.main()
