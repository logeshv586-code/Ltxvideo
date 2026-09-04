"""Memory-aware LTX video generator for RTX 4050 laptops."""
from __future__ import annotations

import gc
import inspect
import random
from datetime import datetime
from pathlib import Path
from typing import Callable

import torch
from PIL import Image

from config import (
    CPU_MEMORY_BUDGET,
    DEFAULT_FPS,
    ENABLE_8BIT,
    ENABLE_VAE_TILING,
    GPU_MEMORY_BUDGET,
    HF_CACHE_DIR,
    HF_REPO_ID,
    MAX_NATIVE_FRAMES,
    MODELS_DIR,
    OUTPUTS_DIR,
)
from engine.skill_engine import SKILL_ENGINE, SkillPlan, VideoRequest
from engine.video_qc import VideoQCReport, inspect_video

Progress = Callable[[str, float], None] | None


def is_fatal_cuda_error(exc: BaseException) -> bool:
    """Whether CUDA can no longer safely run another inference in this process.

    Error 700 is different from an out-of-memory condition. CUDA marks the
    device context as sticky after an illegal address, so cache clearing or an
    image-to-video fallback can merely report the same failure later.
    """
    message = str(exc).lower()
    markers = (
        "cudaerrorillegaladdress",
        "cuda_error_illegal_address",
        "cuda error: an illegal memory access",
        "sticky error detected",
        "returning 700",
        "cuda error 700",
    )
    return any(marker in message for marker in markers)


class VideoGenerator:
    """LTX text/image/video-conditioned generation with memory control."""

    def __init__(self) -> None:
        self.pipe = None
        self.mode: str | None = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.last_skill_plan: SkillPlan | None = None
        self.last_qc: VideoQCReport | None = None

    def _report(self, callback: Progress, message: str, progress: float) -> None:
        print(f"[{progress * 100:5.1f}%] {message}")
        if callback:
            callback(message, progress)

    @staticmethod
    def _validate_shape(width: int, height: int, num_frames: int) -> None:
        if width % 32 or height % 32:
            raise ValueError("LTX width and height must be divisible by 32.")
        if (num_frames - 1) % 8:
            raise ValueError("LTX frame count must follow 8k+1 (49, 97, 193, 241, ...).")
        if num_frames > MAX_NATIVE_FRAMES:
            raise ValueError(
                f"RTX 4050 mode caps one native render at {MAX_NATIVE_FRAMES} frames. "
                "Use Continuous Video mode for longer output."
            )

    def prepare_skill_plan(
        self,
        prompt: str,
        negative_prompt: str,
        width: int,
        height: int,
        num_frames: int,
        has_reference: bool,
        skill_mode: str = "auto",
        character_lock: str = "",
        fps: int = DEFAULT_FPS,
    ) -> SkillPlan:
        fps = max(1, int(fps))
        plan = SKILL_ENGINE.plan(
            VideoRequest(
                raw_prompt=prompt,
                mode=skill_mode,
                duration_seconds=num_frames / fps,
                width=width,
                height=height,
                num_frames=num_frames,
                has_reference=has_reference,
                negative_prompt=negative_prompt,
                character_lock=character_lock,
            )
        )
        self.last_skill_plan = plan
        return plan

    def _load_pipeline(self, mode: str, progress_callback: Progress = None) -> None:
        if self.pipe is not None and self.mode == mode:
            return
        self.unload_model()

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU not detected. An NVIDIA RTX GPU is required.")

        self._report(progress_callback, "Loading low-memory LTX pipeline…", 0.05)
        try:
            from diffusers import (
                BitsAndBytesConfig as DiffusersBitsAndBytesConfig,
                LTXConditionPipeline,
                LTXImageToVideoPipeline,
                LTXPipeline,
                LTXVideoTransformer3DModel,
            )
            from transformers import BitsAndBytesConfig, T5EncoderModel
        except ImportError as exc:
            raise RuntimeError(
                "Required LTX packages are missing. Run `pip install -r requirements.txt` and restart."
            ) from exc

        if not ENABLE_8BIT:
            raise RuntimeError("RTX 4050 mode expects 8-bit loading to fit 6 GB VRAM.")

        # Retained as explicit hardware budgets for diagnostics. The stable path
        # uses model CPU offload instead of a fragile per-module device map.
        _ = {0: GPU_MEMORY_BUDGET, "cpu": CPU_MEMORY_BUDGET}
        (MODELS_DIR / "offload").mkdir(parents=True, exist_ok=True)
        cache_dir = str(HF_CACHE_DIR)

        self._report(progress_callback, "Loading text encoder in 8-bit…", 0.12)
        text_encoder = T5EncoderModel.from_pretrained(
            HF_REPO_ID,
            subfolder="text_encoder",
            cache_dir=cache_dir,
            quantization_config=BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_enable_fp32_cpu_offload=True,
            ),
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )

        self._report(progress_callback, "Loading video transformer in 8-bit…", 0.24)
        transformer = LTXVideoTransformer3DModel.from_pretrained(
            HF_REPO_ID,
            subfolder="transformer",
            cache_dir=cache_dir,
            quantization_config=DiffusersBitsAndBytesConfig(load_in_8bit=True),
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )

        if mode == "i2v":
            pipeline_cls = LTXImageToVideoPipeline
        elif mode == "condition":
            pipeline_cls = LTXConditionPipeline
        else:
            pipeline_cls = LTXPipeline

        self._report(progress_callback, f"Building {mode.upper()} pipeline…", 0.38)
        self.pipe = pipeline_cls.from_pretrained(
            HF_REPO_ID,
            cache_dir=cache_dir,
            text_encoder=text_encoder,
            transformer=transformer,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )
        self.pipe.enable_model_cpu_offload()

        if ENABLE_VAE_TILING and hasattr(self.pipe.vae, "enable_tiling"):
            self.pipe.vae.enable_tiling()
        if hasattr(self.pipe.vae, "enable_slicing"):
            self.pipe.vae.enable_slicing()
        self.mode = mode
        self._report(progress_callback, "LTX pipeline ready.", 0.45)

    def unload_model(self) -> None:
        if self.pipe is not None:
            del self.pipe
        self.pipe = None
        self.mode = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass

    @staticmethod
    def _seed(seed: int) -> int:
        return random.randint(0, 2**31 - 1) if seed is None or int(seed) < 0 else int(seed)

    @staticmethod
    def _output_path(prefix: str) -> Path:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        return OUTPUTS_DIR / f"{prefix}_{stamp}.mp4"

    def _run_pipe_with_cuda_retry(self, mode: str, kwargs: dict, progress_callback: Progress):
        self._load_pipeline(mode, progress_callback)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        run_kwargs = self._with_step_progress(kwargs, progress_callback)
        try:
            result = self.pipe(**run_kwargs)
            self._report(progress_callback, "Denoising complete; decoding video frames…", 0.91)
            return result
        except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
            message = str(exc).lower()
            memory_error = isinstance(exc, torch.cuda.OutOfMemoryError) or "out of memory" in message
            if not memory_error:
                raise
            self._report(progress_callback, "GPU memory pressure detected; retrying once…", 0.46)
            self.unload_model()
            self._load_pipeline(mode, progress_callback)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            result = self.pipe(**self._with_step_progress(kwargs, progress_callback))
            self._report(progress_callback, "Denoising complete; decoding video frames…", 0.91)
            return result

    def _with_step_progress(self, kwargs: dict, progress_callback: Progress) -> dict:
        """Attach a Diffusers denoising callback when the loaded pipeline supports it.

        Without this callback LTX can spend several minutes in ``pipe(...)``
        after reporting only "Generating".  That looks like a frozen T4 job,
        even though the GPU is working normally.  The callback is deliberately
        tensor-free: it only updates UI progress and never copies GPU data.
        """
        if self.pipe is None:
            return kwargs

        try:
            parameters = inspect.signature(self.pipe.__call__).parameters
            supports_callback = "callback_on_step_end" in parameters or any(
                parameter.kind is inspect.Parameter.VAR_KEYWORD
                for parameter in parameters.values()
            )
        except (TypeError, ValueError):
            supports_callback = False
            parameters = {}
        run_kwargs = dict(kwargs)
        # The bundled T5 encoder supports 128 tokens. A larger Diffusers
        # default can turn a long character bible into an indexing failure.
        if "max_sequence_length" in parameters and "max_sequence_length" not in run_kwargs:
            run_kwargs["max_sequence_length"] = 128

        if progress_callback is None:
            return run_kwargs

        if not supports_callback:
            self._report(
                progress_callback,
                "Rendering video (this Diffusers version cannot report individual steps)…",
                0.5,
            )
            return run_kwargs

        total_steps = max(1, int(kwargs.get("num_inference_steps", 1)))

        def on_step_end(_pipe, step: int, _timestep, callback_kwargs: dict):
            completed = min(total_steps, int(step) + 1)
            value = 0.5 + 0.4 * completed / total_steps
            self._report(
                progress_callback,
                f"Rendering video · diffusion step {completed}/{total_steps}",
                value,
            )
            return callback_kwargs

        run_kwargs["callback_on_step_end"] = on_step_end
        run_kwargs["callback_on_step_end_tensor_inputs"] = []
        return run_kwargs

    def _record_qc(
        self,
        path: Path,
        width: int,
        height: int,
        num_frames: int,
        fps: int = DEFAULT_FPS,
    ) -> VideoQCReport:
        report = inspect_video(
            path,
            expected_width=width,
            expected_height=height,
            expected_duration=num_frames / max(1, int(fps)),
        )
        self.last_qc = report
        return report

    def generation_report(self) -> str:
        parts: list[str] = []
        if self.last_skill_plan:
            parts.append(self.last_skill_plan.trace_text())
        if self.last_qc:
            parts.append(self.last_qc.summary())
        return "\n".join(parts)

    def generate_text_to_video(
        self,
        prompt: str,
        negative_prompt: str,
        width: int,
        height: int,
        num_frames: int,
        num_inference_steps: int,
        guidance_scale: float,
        seed: int = -1,
        progress_callback: Progress = None,
        skill_mode: str = "auto",
        character_lock: str = "",
        fps: int = DEFAULT_FPS,
    ) -> Path:
        from diffusers.utils import export_to_video

        self._validate_shape(width, height, num_frames)
        plan = self.prepare_skill_plan(
            prompt,
            negative_prompt,
            width,
            height,
            num_frames,
            has_reference=False,
            skill_mode=skill_mode,
            character_lock=character_lock,
            fps=fps,
        )
        actual_seed = self._seed(seed)
        self._report(progress_callback, f"Generating opening clip (seed {actual_seed})…", 0.5)
        kwargs = dict(
            prompt=plan.prompt,
            negative_prompt=plan.negative_prompt,
            width=width,
            height=height,
            num_frames=num_frames,
            frame_rate=int(fps),
            num_inference_steps=int(num_inference_steps),
            guidance_scale=float(guidance_scale),
            decode_timestep=0.05,
            decode_noise_scale=0.025,
            generator=torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu").manual_seed(actual_seed),
        )
        frames = self._run_pipe_with_cuda_retry("t2v", kwargs, progress_callback).frames[0]
        self._report(progress_callback, "Encoding MP4…", 0.95)
        output = self._output_path("ltx_t2v")
        export_to_video(frames, str(output), fps=int(fps))
        self._report(progress_callback, "Checking generated video…", 0.98)
        qc = self._record_qc(output, width, height, num_frames, fps=fps)
        self._report(progress_callback, f"Saved {output.name} · QC {'PASS' if not qc.fatal else 'FAIL'}", 1.0)
        return output

    def generate_image_to_video(
        self,
        prompt: str,
        image: Image.Image,
        negative_prompt: str,
        width: int,
        height: int,
        num_frames: int,
        num_inference_steps: int,
        guidance_scale: float,
        seed: int = -1,
        progress_callback: Progress = None,
        skill_mode: str = "auto",
        character_lock: str = "",
        fps: int = DEFAULT_FPS,
    ) -> Path:
        from diffusers.utils import export_to_video

        self._validate_shape(width, height, num_frames)
        if image is None:
            raise ValueError("Reference image is required for image-to-video.")
        plan = self.prepare_skill_plan(
            prompt,
            negative_prompt,
            width,
            height,
            num_frames,
            has_reference=True,
            skill_mode=skill_mode,
            character_lock=character_lock,
            fps=fps,
        )
        actual_seed = self._seed(seed)
        prepared = image.convert("RGB").resize((width, height), Image.Resampling.LANCZOS)
        self._report(progress_callback, f"Animating reference image (seed {actual_seed})…", 0.5)
        kwargs = dict(
            image=prepared,
            prompt=plan.prompt,
            negative_prompt=plan.negative_prompt,
            width=width,
            height=height,
            num_frames=num_frames,
            frame_rate=int(fps),
            num_inference_steps=int(num_inference_steps),
            guidance_scale=float(guidance_scale),
            decode_timestep=0.05,
            decode_noise_scale=0.025,
            generator=torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu").manual_seed(actual_seed),
        )
        frames = self._run_pipe_with_cuda_retry("i2v", kwargs, progress_callback).frames[0]
        self._report(progress_callback, "Encoding MP4…", 0.95)
        output = self._output_path("ltx_i2v")
        export_to_video(frames, str(output), fps=int(fps))
        self._report(progress_callback, "Checking generated video…", 0.98)
        qc = self._record_qc(output, width, height, num_frames, fps=fps)
        self._report(progress_callback, f"Saved {output.name} · QC {'PASS' if not qc.fatal else 'FAIL'}", 1.0)
        return output

    def generate_conditioned_video(
        self,
        prompt: str,
        negative_prompt: str,
        width: int,
        height: int,
        num_frames: int,
        num_inference_steps: int,
        guidance_scale: float,
        seed: int = -1,
        progress_callback: Progress = None,
        character_lock: str = "",
        fps: int = 24,
        conditioning_frames: list[Image.Image] | None = None,
        reference_image: Image.Image | None = None,
        condition_strength: float = 1.0,
        image_cond_noise_scale: float = 0.025,
    ) -> Path:
        """Generate or extend a clip with the official LTX condition pipeline.

        The direct ``video=`` / ``image=`` API is used instead of constructing
        internal LTXVideoCondition objects. This is both simpler and compatible
        with Diffusers 0.37.x, which is the version range used by this project.
        """
        from diffusers.utils import export_to_video

        self._validate_shape(width, height, num_frames)
        fps = max(1, int(fps))
        has_reference = bool(conditioning_frames) or reference_image is not None
        plan = self.prepare_skill_plan(
            prompt,
            negative_prompt,
            width,
            height,
            num_frames,
            has_reference=has_reference,
            skill_mode="auto",
            character_lock=character_lock,
            fps=fps,
        )

        kwargs = dict(
            prompt=plan.prompt,
            negative_prompt=plan.negative_prompt,
            width=width,
            height=height,
            num_frames=num_frames,
            frame_rate=fps,
            num_inference_steps=int(num_inference_steps),
            guidance_scale=float(guidance_scale),
            guidance_rescale=0.7,
            image_cond_noise_scale=float(image_cond_noise_scale),
            decode_timestep=0.05,
            decode_noise_scale=0.025,
            generator=torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu").manual_seed(self._seed(seed)),
        )

        prefix = "ltx_condition"
        if conditioning_frames:
            prepared_video = [
                frame.convert("RGB").resize((width, height), Image.Resampling.LANCZOS)
                for frame in conditioning_frames
            ]
            kwargs.update(
                video=prepared_video,
                frame_index=0,
                strength=float(condition_strength),
            )
            prefix = "ltx_continue"
            self._report(progress_callback, "Continuing from previous motion frames…", 0.5)
        elif reference_image is not None:
            kwargs.update(
                image=reference_image.convert("RGB").resize((width, height), Image.Resampling.LANCZOS),
                frame_index=0,
                strength=float(condition_strength),
            )
            self._report(progress_callback, "Animating reference image…", 0.5)
        else:
            self._report(progress_callback, "Generating conditioned clip…", 0.5)

        frames = self._run_pipe_with_cuda_retry("condition", kwargs, progress_callback).frames[0]
        self._report(progress_callback, "Encoding MP4…", 0.95)
        output = self._output_path(prefix)
        export_to_video(frames, str(output), fps=fps)
        self._report(progress_callback, "Checking generated video…", 0.98)
        qc = self._record_qc(output, width, height, num_frames, fps=fps)
        state = "PASS" if not qc.fatal and not qc.visual_failure else "REVIEW"
        self._report(progress_callback, f"Saved {output.name} · QC {state}", 1.0)
        return output
