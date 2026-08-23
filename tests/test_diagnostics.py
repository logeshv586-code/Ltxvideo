from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

import diagnostics


class _FakeCuda:
    def __init__(self, name: str = "NVIDIA GeForce RTX 4050 Laptop GPU", vram_gb: float = 6.0) -> None:
        self.name = name
        self.vram_gb = vram_gb

    @staticmethod
    def is_available() -> bool:
        return True

    def get_device_name(self, index: int) -> str:
        return self.name

    def get_device_properties(self, index: int):
        return SimpleNamespace(total_memory=int(self.vram_gb * 1024**3))


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

    def _ready_results(self, cuda: _FakeCuda):
        fake_torch = SimpleNamespace(
            __version__="2.7.0",
            cuda=cuda,
            version=SimpleNamespace(cuda="12.8"),
        )
        fake_ffmpeg = SimpleNamespace(get_ffmpeg_exe=lambda: "C:/ffmpeg/ffmpeg.exe")

        def fake_loader(name: str):
            return fake_torch if name == "torch" else fake_ffmpeg

        with (
            patch("diagnostics.importlib.util.find_spec", return_value=object()),
            patch.object(Path, "exists", return_value=True),
        ):
            return diagnostics.collect_diagnostics(loader=fake_loader, marker=Path("models/.ltx_ready"))

    def test_ready_rtx4050_environment_passes(self) -> None:
        results = self._ready_results(_FakeCuda())

        self.assertEqual(diagnostics.exit_code(results), 0)
        self.assertTrue(any("RTX 4050" in r.detail for r in results if r.name == "CUDA"))
        self.assertTrue(any("RTX 4050 6 GB" in r.detail for r in results if r.name == "GPU profile"))
        self.assertTrue(any(r.name == "FFmpeg" and r.level == "PASS" for r in results))

    def test_rtx3050_4gb_uses_safe_profile(self) -> None:
        results = self._ready_results(_FakeCuda("NVIDIA GeForce RTX 3050 Laptop GPU", 4.0))

        self.assertEqual(diagnostics.exit_code(results), 0)
        self.assertTrue(any("RTX 3050 4 GB safe profile" in r.detail for r in results if r.name == "GPU profile"))
        self.assertTrue(any("native cap 121" in r.detail for r in results if r.name == "Safe preset"))
