from unittest import TestCase

from engine.hardware_profiles import select_hardware_profile


class HardwareProfileTests(TestCase):
    def test_rtx3050_4gb_profile_reserves_vram(self) -> None:
        profile = select_hardware_profile("NVIDIA GeForce RTX 3050 Laptop GPU", 4.0, 16.0)
        self.assertEqual(profile.key, "rtx3050-4gb")
        self.assertEqual(profile.gpu_memory_budget, "3GiB")
        self.assertEqual(profile.max_native_frames, 121)
        self.assertEqual(profile.safe_preset, "384x224 · 49 frames")

    def test_rtx3050_6gb_profile_is_less_restrictive(self) -> None:
        profile = select_hardware_profile("NVIDIA GeForce RTX 3050 Laptop GPU", 6.0, 16.0)
        self.assertEqual(profile.key, "rtx3050-6gb")
        self.assertEqual(profile.gpu_memory_budget, "4GiB")
        self.assertEqual(profile.max_native_frames, 193)

    def test_rtx4050_keeps_existing_budget(self) -> None:
        profile = select_hardware_profile("NVIDIA GeForce RTX 4050 Laptop GPU", 6.0, 16.0)
        self.assertEqual(profile.key, "rtx4050-6gb")
        self.assertEqual(profile.gpu_memory_budget, "5GiB")
        self.assertEqual(profile.max_native_frames, 241)

    def test_generic_4gb_gpu_uses_safe_fallback(self) -> None:
        profile = select_hardware_profile("NVIDIA CUDA GPU", 4.0, 16.0)
        self.assertEqual(profile.key, "generic-4gb")
        self.assertEqual(profile.max_native_frames, 121)
