import unittest

from engine.generator import is_fatal_cuda_error
from engine.optimized_generator import _OptimizedWorker


class CudaRecoveryTests(unittest.TestCase):
    def test_illegal_address_is_not_treated_as_an_ordinary_fallback(self):
        error = RuntimeError(
            "CUDA error: an illegal memory access was encountered; "
            "Sticky error detected Returning 700"
        )
        self.assertTrue(is_fatal_cuda_error(error))
        self.assertFalse(is_fatal_cuda_error(RuntimeError("CUDA out of memory")))

    def test_conditioning_never_receives_adaptive_text_upscale(self):
        worker = object.__new__(_OptimizedWorker)
        kwargs = {"width": 576, "height": 320, "num_frames": 113}
        adapted = worker._adapt_kwargs(kwargs, "condition")
        self.assertEqual((adapted["width"], adapted["height"]), (576, 320))

    def test_longform_can_disable_text_upscale_for_matching_clip_sizes(self):
        worker = object.__new__(_OptimizedWorker)
        kwargs = {
            "width": 576,
            "height": 320,
            "num_frames": 97,
            "_adaptive_native_upscale": False,
        }
        adapted = worker._adapt_kwargs(kwargs, "t2v")
        self.assertEqual((adapted["width"], adapted["height"]), (576, 320))
        self.assertNotIn("_adaptive_native_upscale", adapted)


if __name__ == "__main__":
    unittest.main()
