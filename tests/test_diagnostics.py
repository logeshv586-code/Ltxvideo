from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

import diagnostics


class _FakeCuda:
    @staticmethod
    def is_available() -> bool:
        return True

    @staticmethod
    def get_device_name(index: int) -> str:
        return "NVIDIA GeForce RTX 4050 Laptop GPU"


class DiagnosticsTests(TestCase):
    def test_missing_torch_is_blocking_but_missing_cache_is_not(self) -> None:
        with (
            patch("diagnostics.importlib.util.find_spec", return_value=None),
            patch("diagnostics.shutil.which", return_value=None),
        ):
            results = diagnostics.collect_diagnostics(marker=Path("definitely-missing-marker"))

        self.assertEqual(diagnostics.exit_code(results), 1)
        self.assertTrue(any(r.name == "PyTorch" and r.level == "FAIL" for r in results))
        self.assertTrue(any(r.name == "Model cache" and r.level == "INFO" for r in results))

    def test_ready_rtx4050_environment_passes(self) -> None:
        fake_torch = SimpleNamespace(
            __version__="2.7.0",
            cuda=_FakeCuda(),
            version=SimpleNamespace(cuda="12.8"),
        )
        fake_ffmpeg = SimpleNamespace(get_ffmpeg_exe=lambda: "C:/ffmpeg/ffmpeg.exe")

        def fake_loader(name: str):
            return fake_torch if name == "torch" else fake_ffmpeg

        with (
            patch("diagnostics.importlib.util.find_spec", return_value=object()),
            patch.object(Path, "exists", return_value=True),
        ):
            results = diagnostics.collect_diagnostics(loader=fake_loader, marker=Path("models/.ltx_ready"))

        self.assertEqual(diagnostics.exit_code(results), 0)
        self.assertTrue(any("RTX 4050" in r.detail for r in results if r.name == "CUDA"))
        self.assertTrue(any(r.name == "FFmpeg" and r.level == "PASS" for r in results))
