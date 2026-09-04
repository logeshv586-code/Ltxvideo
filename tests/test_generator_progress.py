from types import SimpleNamespace
from unittest import TestCase

from engine.generator import VideoGenerator


class _CallbackPipe:
    def __call__(
        self,
        *,
        num_inference_steps: int,
        callback_on_step_end,
        callback_on_step_end_tensor_inputs,
        **_kwargs,
    ):
        if callback_on_step_end_tensor_inputs != []:
            raise AssertionError("Progress callbacks must not request GPU tensors")
        for step in range(num_inference_steps):
            callback_on_step_end(self, step, step, {})
        return SimpleNamespace(frames=[[]])


class GeneratorProgressTests(TestCase):
    def test_pipeline_progress_reports_each_denoising_step(self) -> None:
        generator = VideoGenerator()
        generator._load_pipeline = lambda _mode, _callback: setattr(generator, "pipe", _CallbackPipe())
        events: list[tuple[str, float]] = []

        generator._run_pipe_with_cuda_retry(
            "t2v",
            {"num_inference_steps": 3},
            lambda message, value: events.append((message, value)),
        )

        step_messages = [message for message, _value in events if "diffusion step" in message]
        self.assertEqual(len(step_messages), 3)
        self.assertIn("diffusion step 1/3", step_messages[0])
        self.assertIn("diffusion step 3/3", step_messages[-1])
        self.assertEqual(events[-1][1], 0.91)

