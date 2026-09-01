"""Memory-aware LTX-Video generator for RTX 4050 laptops.

Every generation passes through the Video Skill Engine before inference. The
engine preserves the user's literal intent, applies only relevant directing
skills, hardens negative constraints, and performs a post-render technical QC.

Long-form generation uses LTXConditionPipeline so a short sequence of previous
frames can condition the next shot. This avoids the recursive last-frame-only
I2V degradation that caused geometry and sharpness drift in earlier builds.
"""
from __future__ import annotations

import gc
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
            raise ValueError("LTX frame count must follow 8k+1 (49, 97, 121, ...).")
        if num_frames > MAX_NATIVE_FRAMES:
            raise ValueError(
                f"RTX 4050 mode caps one native clip at {MAX_NATIVE_FRAMES} frames. "
                "Use Easy Video Creator for longer videos."
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
        """Build the mandatory skill plan used by every generation path."""
        fps = max(1, int(fps))
        plan = SKILL_ENGINE.plan(VideoRequest(
            raw_prompt=prompt,
            mode=skill_mode,
            duration_seconds=num_frames / fps,
            width=width,
            height=height,
            num_frames=num_frames,
            has_reference=has_reference,
            negative_prompt=negative_prompt,
            character_lock=character_lock,
        ))
        self.last_skill_plan = plan
        return plan

    def _load_pipeline(self, mode: str, progress_callback: Progress = None) -> None:
        if self.pipe is not None and self.mode == mode:
            return
        self.unload_model()

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU not detected. This profile requires an NVIDIA RTX GPU.")

        self._report(progress_callback, "Loading quantized LTX 2B pipeline…", 0.05)

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
                "Missing LTX runtime packages. Run `python run.py` once to install dependencies."
            ) from exc

        if not ENABLE_8BIT:
            raise RuntimeError("This build expects 8-bit mode for RTX 4050 hardware.")

        max_memory = {0: GPU_MEMORY_BUDGET, "cpu": CPU_MEMORY_BUDGET}
        # Keep these values materialized for diagnostics and future device-map
        # support. CPU offload below remains the stable Windows/6 GB path.
        _ = max_memory
        offload_folder = str(MODELS_DIR / "offload")
        Path(offload_folder).mkdir(parents=True, exist_ok=True)
        cache_dir = str(HF_CACHE_DIR)

        self._report(progress_callback, "Loading T5 text encoder in 8-bit…", 0.12)
        text_encoder = T5EncoderModel.from_pretrained(
            HF_REPO_ID,
            subfolder="text_encoder",
            cache_dir=cache_dir,
            quantization_config=BitsAndBytesConfig(load_in_8bit=True, llm_int8_enable_fp32_cpu_offload=True),
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )

        self._report(progress_callback, "Loading 2B video transformer in 8-bit…", 0.24)
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
        try:
            return self.pipe(**kwargs)
        except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
            message = str(exc).lower()
            if not isinstance(exc, torch.cuda.OutOfMemoryError) and "cuda" not in message and "out of memory" not in message:
                raise
            self._report(progress_callback, "CUDA memory failure detected; clearing the model and retrying once…", 0.46)
            self.unload_model()
            self._load_pipeline(mode, progress_callback)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return self.pipe(**kwargs)

    def _record_qc(
        self,
        path: Path,
        width: int,
        height: int,
        num_frames: int,
        fps: int = DEFAULT_FPS,
    ) -> VideoQCReport:
        fps = max(1, int(fps))
        report = inspect_video(
            path,
            expected_width=width,
            expected_height=height,
            expected_duration=num_frames / fps,
        )
        self.last_qc = report
        return report

    def generation_report(self) -> str:
        """Human-readable skill + technical QC trace for the UI/logs."""
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
            prompt, negative_prompt, width, height, num_frames,
            has_reference=False, skill_mode=skill_mode, character_lock=character_lock, fps=fps,
        )
        self._report(
            progress_callback,
            f"Video Skill Engine: {len(plan.applied_skills)} skills active · quality gate {plan.quality_score}/100",
            0.02,
        )
        actual_seed = self._seed(seed)
        self._report(progress_callback, f"Generating {num_frames} frames (seed {actual_seed})…", 0.5)
        generator = torch.Generator(device="cuda").manual_seed(actual_seed)
        kwargs = dict(
            prompt=plan.prompt,
            negative_prompt=plan.negative_prompt,
            width=width,
            height=height,
            num_frames=num_frames,
            num_inference_steps=int(num_inference_steps),
            guidance_scale=float(guidance_scale),
            decode_timestep=0.05,
            decode_noise_scale=0.025,
            generator=generator,
        )
        result = self._run_pipe_with_cuda_retry("t2v", kwargs, progress_callback).frames[0]
        output = self._output_path("ltx_t2v")
        export_to_video(result, str(output), fps=int(fps))
        qc = self._record_qc(output, width, height, num_frames, fps=fps)
        self._report(progress_callback, f"Saved {output.name} · technical QC {'PASS' if not qc.fatal else 'FAIL'}", 1.0)
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
            prompt, negative_prompt, width, height, num_frames,
            has_reference=True, skill_mode=skill_mode, character_lock=character_lock, fps=fps,
        )
        self._report(
            progress_callback,
            f"Video Skill Engine: {len(plan.applied_skills)} skills active · quality gate {plan.quality_score}/100",
            0.02,
        )
        actual_seed = self._seed(seed)
        image = image.convert("RGB").resize((width, height), Image.Resampling.LANCZOS)
        self._report(progress_callback, f"Animating reference image (seed {actual_seed})…", 0.5)
        generator = torch.Generator(device="cuda").manual_seed(actual_seed)
        kwargs = dict(
            image=image,
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
            generator=generator,
        )
        result = self._run_pipe_with_cuda_retry("i2v", kwargs, progress_callback).frames[0]
        output = self._output_path("ltx_i2v")
        export_to_video(result, str(output), fps=int(fps))
        qc = self._record_qc(output, width, height, num_frames, fps=fps)
        self._report(progress_callback, f"Saved {output.name} · technical QC {'PASS' if not qc.fatal else 'FAIL'}", 1.0)
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
        image_cond_noise_scale: float = 0.05,
    ) -> Path:
        """Generate/extend a clip using LTX multi-frame conditioning.

        ``conditioning_frames`` takes priority over ``reference_image``. The
        long-form renderer passes the previous clip's final motion sequence
        here, allowing LTX to observe movement rather than guessing motion from
        one potentially degraded still frame.
        """
        from diffusers.pipelines.ltx.pipeline_ltx_condition import LTXVideoCondition
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
        self._report(
            progress_callback,
            f"Video Skill Engine: {len(plan.applied_skills)} skills active · quality gate {plan.quality_score}/100",
            0.02,
        )

        conditions = []
        if conditioning_frames:
            prepared = [
                frame.convert("RGB").resize((width, height), Image.Resampling.LANCZOS)
                for frame in conditioning_frames
            ]
            conditions.append(LTXVideoCondition(
                video=prepared,
                frame_index=0,
                strength=float(condition_strength),
            ))
        elif reference_image is not None:
            prepared_image = reference_image.convert("RGB").resize(
                (width, height), Image.Resampling.LANCZOS
            )
            conditions.append(LTXVideoCondition(
                image=prepared_image,
                frame_index=0,
                strength=float(condition_strength),
            ))

        actual_seed = self._seed(seed)
        action = "Extending previous motion" if conditioning_frames else (
            "Animating reference image" if reference_image is not None else "Generating opening shot"
        )
        self._report(progress_callback, f"{action} (seed {actual_seed})…", 0.5)
        generator = torch.Generator(device="cuda").manual_seed(actual_seed)
        kwargs = dict(
            prompt=plan.prompt,
            negative_prompt=plan.negative_prompt,
            width=width,
            height=height,
            num_frames=num_frames,
            frame_rate=fps,
            num_inference_steps=int(num_inference_steps),
            guidance_scale=float(guidance_scale),
            image_cond_noise_scale=float(image_cond_noise_scale),
            decode_timestep=0.05,
            decode_noise_scale=0.025,
            generator=generator,
        )
        if conditions:
            kwargs["conditions"] = conditions

        result = self._run_pipe_with_cuda_retry("condition", kwargs, progress_callback).frames[0]
        prefix = "ltx_continue" if conditioning_frames else "ltx_condition"
        output = self._output_path(prefix)
        export_to_video(result, str(output), fps=fps)
        qc = self._record_qc(output, width, height, num_frames, fps=fps)
        state = "PASS" if not qc.fatal and not qc.visual_failure else "REVIEW"
        self._report(progress_callback, f"Saved {output.name} · scene QC {state}", 1.0)
        return output
