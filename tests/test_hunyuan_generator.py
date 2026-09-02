from __future__ import annotations

import unittest
from pathlib import Path

from engine.hunyuan_generator import HunyuanVideoGenerator


class HunyuanGeneratorTests(unittest.TestCase):
    def setUp(self):
        self.generator = HunyuanVideoGenerator(
            source_dir=Path("third_party/HunyuanVideo-1.5"),
            model_dir=Path("models/hunyuanvideo-1.5"),
        )

    def test_step_distilled_command_uses_safe_4080_defaults(self):
        command = self.generator.build_command(
            prompt="Moon Cookie waves at the camera",
            image_path=Path("reference.png"),
            output_path=Path("output.mp4"),
            steps=12,
            enable_sr=True,
            overlap_group_offloading=False,
        )
        joined = " ".join(map(str, command))
        self.assertIn("--resolution 480p", joined)
        self.assertIn("--enable_step_distill true", joined)
        self.assertIn("--num_inference_steps 12", joined)
        self.assertIn("--enable_cache false", joined)
        self.assertIn("--offloading true", joined)
        self.assertIn("--group_offloading true", joined)
        self.assertIn("--overlap_group_offloading false", joined)
        self.assertIn("--sr true", joined)

    def test_rejects_non_distilled_step_count(self):
        with self.assertRaises(ValueError):
            self.generator.build_command(
                prompt="test",
                image_path=Path("reference.png"),
                output_path=Path("output.mp4"),
                steps=20,
            )


if __name__ == "__main__":
    unittest.main()
