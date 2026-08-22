from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools.capture_rtx4050_proof import build_manifest, sha256_file
from tools.check_repo_growth import REQUIRED_TOPICS, evaluate


class DemoProofTests(unittest.TestCase):
    def test_hash_and_4050_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            clip = root / "clip.mp4"
            clip.write_bytes(b"demo-bytes")
            gpu = {
                "available": "true",
                "name": "NVIDIA GeForce RTX 4050 Laptop GPU",
                "driver": "999.1",
                "memory_mb": "6141",
            }
            with patch("tools.capture_rtx4050_proof.git_commit", return_value="abc123"):
                manifest = build_manifest(root, [clip], gpu)
            self.assertEqual(manifest["git_commit"], "abc123")
            self.assertEqual(len(manifest["shots"]), 4)
            self.assertEqual(manifest["clips"][0]["sha256"], sha256_file(clip))

    def test_non_4050_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(RuntimeError):
                build_manifest(
                    Path(tmp),
                    [],
                    {"available": "true", "name": "NVIDIA GeForce RTX 4090"},
                )

    def test_repo_growth_evaluation(self):
        checks = evaluate(
            {
                "description": "Local LTX video studio",
                "topics": sorted(REQUIRED_TOPICS),
                "has_discussions": True,
            }
        )
        self.assertTrue(all(ok for _, ok, _ in checks))

    def test_repo_growth_detects_missing_settings(self):
        checks = evaluate({"description": None, "topics": [], "has_discussions": False})
        self.assertFalse(all(ok for _, ok, _ in checks))


class ConfigureRepoTests(unittest.TestCase):
    def test_configuration_constants(self):
        from tools.configure_repo_settings import DESCRIPTION, TOPICS
        self.assertIn("RTX 4050", DESCRIPTION)
        self.assertEqual(len(TOPICS), 10)
        self.assertIn("ltx-video", TOPICS)


if __name__ == "__main__":
    unittest.main()
