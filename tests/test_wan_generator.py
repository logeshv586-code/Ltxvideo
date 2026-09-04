import unittest

from engine.wan_generator import (
    WAN_FPS,
    WAN_FRAMES,
    WAN_HEIGHT,
    WAN_WIDTH,
    select_wan_offload_mode,
)


class WanGeneratorTests(unittest.TestCase):
    def test_uses_the_official_stable_480p_shape(self):
        self.assertEqual((WAN_WIDTH, WAN_HEIGHT), (832, 480))
        self.assertEqual((WAN_FRAMES, WAN_FPS), (81, 16))

    def test_six_gb_gpu_uses_sequential_cpu_offload(self):
        self.assertEqual(select_wan_offload_mode(6.0, "auto"), "sequential")

    def test_t4_can_use_faster_model_cpu_offload(self):
        self.assertEqual(select_wan_offload_mode(14.6, "auto"), "model")

    def test_explicit_safe_mode_wins_over_hardware_detection(self):
        self.assertEqual(select_wan_offload_mode(14.6, "sequential"), "sequential")


if __name__ == "__main__":
    unittest.main()
