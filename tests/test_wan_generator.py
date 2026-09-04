import unittest

from engine.wan_generator import (
    WAN_CLIP_SECONDS,
    WAN_DELIVERY_FPS,
    WAN_FPS,
    WAN_FRAMES,
    WAN_GGUF_FILENAME,
    WAN_HEIGHT,
    WAN_WIDTH,
    select_wan_backend,
    select_wan_offload_mode,
)


class WanGeneratorTests(unittest.TestCase):
    def test_uses_the_official_stable_480p_shape(self):
        self.assertEqual((WAN_WIDTH, WAN_HEIGHT), (832, 480))
        self.assertEqual((WAN_FRAMES, WAN_FPS), (81, 16))
        self.assertAlmostEqual(WAN_CLIP_SECONDS, 5.0625)

    def test_delivery_uses_two_x_motion_smoothing_target(self):
        self.assertEqual(WAN_DELIVERY_FPS, 32)

    def test_4050_class_gpu_prefers_gguf(self):
        self.assertEqual(select_wan_backend(6.0, "auto"), "gguf")

    def test_t4_class_gpu_prefers_full_weights(self):
        self.assertEqual(select_wan_backend(15.0, "auto"), "full")

    def test_explicit_gguf_override_wins(self):
        self.assertEqual(select_wan_backend(16.0, "gguf"), "gguf")

    def test_default_gguf_targets_quality_balanced_q5(self):
        self.assertTrue(WAN_GGUF_FILENAME.endswith("Q5_0.gguf"))

    def test_six_gb_gpu_uses_sequential_cpu_offload(self):
        self.assertEqual(select_wan_offload_mode(6.0, "auto"), "sequential")

    def test_t4_can_use_faster_model_cpu_offload(self):
        self.assertEqual(select_wan_offload_mode(14.6, "auto"), "model")

    def test_explicit_safe_mode_wins_over_hardware_detection(self):
        self.assertEqual(select_wan_offload_mode(14.6, "sequential"), "sequential")


if __name__ == "__main__":
    unittest.main()
