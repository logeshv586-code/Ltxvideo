"""Wan2.1-T2V-1.3B backend tuned for low-VRAM cartoon generation.

RTX 4050-class GPUs automatically prefer a GGUF-quantized Wan transformer while
keeping the official Diffusers tokenizer, text encoder, scheduler and VAE.
Higher-VRAM GPUs can continue to use the normal Diffusers transformer.

Generation stays at Wan's stable 832x480 / 16 FPS shape. Delivery is enhanced
after diffusion with RIFE motion interpolation and Real-ESRGAN when installed.
"""
from __future__ import annotations

import gc
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Callable

import torch

from config import MODELS_DIR, OUTPUTS_DIR
from engine.cogvideox_generator import build_story_sequence
from engine.video_enhancement import enhance_wan_delivery
from engine.video_processor import concatenate_videos_streaming, export_delivery

Progress = Callable[[str, float], None] | None

# Official Diffusers repository used for the pipeline components and GGUF config.
WAN_REPO_ID = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
WAN_MODEL_DIR = Path(os.getenv("WAN_MODEL_DIR", MODELS_DIR / "wan2.1-t2v-1.3b"))

# GGUF only replaces the diffusion transformer. The tokenizer/text encoder/VAE
# are downloaded once from the official Diffusers repository.
WAN_GGUF_REPO_ID = os.getenv(
    "WAN_GGUF_REPO_ID",
    "samuelchristlie/Wan2.1-T2V-1.3B-GGUF",
)
WAN_GGUF_FILENAME = os.getenv(
    "WAN_GGUF_FILENAME",
    "Wan2.1-T2V-1.3B-Q5_0.gguf",
)
WAN_GGUF_DIR = Path(os.getenv("WAN_GGUF_DIR", MODELS_DIR / "wan2.1-t2v-1.3b-gguf"))
WAN_GGUF_PATH = WAN_GGUF_DIR / WAN_GGUF_FILENAME
WAN_GGUF_BASE_DIR = Path(
    os.getenv("WAN_GGUF_BASE_DIR", MODELS_DIR / "wan2.1-t2v-1.3b-diffusers-base")
)

WAN_WIDTH = 832
WAN_HEIGHT = 480
WAN_FRAMES = 81
WAN_FPS = 16
WAN_DELIVERY_FPS = int(os.getenv("WAN_DELIVERY_FPS", "32"))
WAN_STEPS = int(os.getenv("WAN_STEPS", "40"))
WAN_CLIP_SECONDS = WAN_FRAMES / WAN_FPS


def select_wan_backend(total_vram_gb: float, requested: str | None = None) -> str:
    """Choose GGUF automatically for low-VRAM GPUs.

    ``WAN_BACKEND=gguf`` or ``WAN_BACKEND=full`` overrides hardware detection.
    Auto uses GGUF below 10 GB VRAM and the normal Diffusers transformer above.
    """
    choice = (requested or os.getenv("WAN_BACKEND", "auto")).strip().lower()
    aliases = {
        "gguf": "gguf",
        "q4": "gguf",
        "q5": "gguf",
        "quantized": "gguf",
        "full": "full",
        "fp16": "full",
        "bf16": "full",
    }
    if choice in aliases:
        return aliases[choice]
    return "gguf" if float(total_vram_gb) < 10.0 else "full"


def select_wan_offload_mode(total_vram_gb: float, requested: str | None = None) -> str:
    """Use sequential offload below 12 GB so 6 GB laptops remain usable."""
    choice = (requested or os.getenv("WAN_OFFLOAD_MODE", "auto")).strip().lower()
    if choice in {"model", "sequential"}:
        return choice
    return "sequential" if float(total_vram_gb) < 12.0 else "model"


def _gguf_ready() -> tuple[bool, str]:
    base_required = (
        "model_index.json",
        "scheduler",
        "tokenizer",
        "text_encoder",
        "vae",
        "transformer/config.json",
    )
    missing = [item for item in base_required if not (WAN_GGUF_BASE_DIR / item).exists()]
    if not WAN_GGUF_PATH.exists():
        missing.append(WAN_GGUF_FILENAME)
    if missing:
        return False, f"GGUF setup required; missing: {', '.join(missing)}"
    return True, f"GGUF ready: {WAN_GGUF_FILENAME}"


def _full_ready() -> tuple[bool, str]:
    required = ("model_index.json", "transformer", "text_encoder", "vae")
    missing = [item for item in required if not (WAN_MODEL_DIR / item).exists()]
    if missing:
        return False, f"Full model setup required at {WAN_MODEL_DIR}; missing: {', '.join(missing)}"
    return True, "Full Diffusers model ready"


class WanVideoGenerator:
    """Lazy-loaded Wan pipeline with automatic GGUF selection on low VRAM."""

    def __init__(self, device_index: int = 0) -> None:
        self.device_index = int(device_index)
        self.pipe = None
        self.backend: str | None = None

    @staticmethod
    def _report(callback: Progress, message: str, value: float) -> None:
        if callback:
            callback(message, min(1.0, max(0.0, float(value))))

    @staticmethod
    def _seed(seed: int | float) -> int:
        return random.randint(0, 2**31 - 1) if int(seed) < 0 else int(seed)

    def _clear_cuda_cache(self) -> None:
        gc.collect()
        if torch.cuda.is_available():
            with torch.cuda.device(self.device_index):
                torch.cuda.empty_cache()
                if hasattr(torch.cuda, "ipc_collect"):
                    torch.cuda.ipc_collect()

    def readiness(self) -> tuple[bool, str]:
        """Report the setup appropriate for the current GPU when possible."""
        if torch.cuda.is_available():
            total = torch.cuda.get_device_properties(self.device_index).total_memory / (1024**3)
            backend = select_wan_backend(total)
        else:
            backend = (os.getenv("WAN_BACKEND", "gguf") or "gguf").strip().lower()
            backend = "full" if backend == "full" else "gguf"

        if backend == "gguf":
            ready, detail = _gguf_ready()
            return ready, (
                f"Wan2.1-T2V-1.3B {detail}; native {WAN_WIDTH}x{WAN_HEIGHT} @ {WAN_FPS} FPS, "
                f"delivery {WAN_DELIVERY_FPS} FPS."
            )
        ready, detail = _full_ready()
        return ready, (
            f"Wan2.1-T2V-1.3B {detail}; native {WAN_WIDTH}x{WAN_HEIGHT} @ {WAN_FPS} FPS, "
            f"delivery {WAN_DELIVERY_FPS} FPS."
        )

    def _load_gguf(self, callback: Progress) -> None:
        ready, message = _gguf_ready()
        if not ready:
            raise RuntimeError(f"{message}. Run: python setup_wan.py --download-gguf")

        try:
            from diffusers import (
                GGUFQuantizationConfig,
                WanPipeline,
                WanTransformer3DModel,
            )
        except ImportError as exc:
            raise RuntimeError(
                "GGUF Wan requires a current Diffusers build and the 'gguf' Python package. "
                "Run: pip install -U diffusers gguf"
            ) from exc

        self._report(callback, f"Loading GGUF transformer {WAN_GGUF_FILENAME}", 0.05)
        quant_config = GGUFQuantizationConfig(compute_dtype=torch.float16)
        transformer = WanTransformer3DModel.from_single_file(
            str(WAN_GGUF_PATH),
            quantization_config=quant_config,
            config=str(WAN_GGUF_BASE_DIR),
            subfolder="transformer",
            dtype=torch.float16,
        )
        self.pipe = WanPipeline.from_pretrained(
            str(WAN_GGUF_BASE_DIR),
            transformer=transformer,
            dtype=torch.float16,
            low_cpu_mem_usage=True,
            local_files_only=True,
        )
        self.backend = "gguf"

    def _load_full(self, callback: Progress) -> None:
        ready, message = _full_ready()
        if not ready:
            raise RuntimeError(f"{message}. Run: python setup_wan.py --download")
        try:
            from diffusers import WanPipeline
        except ImportError as exc:
            raise RuntimeError("Wan2.1 needs a current Diffusers installation with WanPipeline support.") from exc

        self._report(callback, "Loading full Wan2.1-T2V-1.3B transformer", 0.05)
        self.pipe = WanPipeline.from_pretrained(
            str(WAN_MODEL_DIR),
            dtype=torch.float16,
            low_cpu_mem_usage=True,
            local_files_only=True,
        )
        self.backend = "full"

    def _load(self, callback: Progress) -> None:
        if self.pipe is not None:
            return
        if not torch.cuda.is_available():
            raise RuntimeError("Wan2.1 requires an NVIDIA CUDA GPU.")

        total_bytes = torch.cuda.get_device_properties(self.device_index).total_memory
        total_vram_gb = total_bytes / (1024**3)
        backend = select_wan_backend(total_vram_gb)
        offload_mode = select_wan_offload_mode(total_vram_gb)

        if backend == "gguf":
            self._load_gguf(callback)
            backend_label = f"GGUF {WAN_GGUF_FILENAME}"
        else:
            self._load_full(callback)
            backend_label = "full Diffusers weights"

        # Wan's VAE is more reliable in fp32. It is comparatively small and
        # remains compatible with CPU offload.
        if os.getenv("WAN_VAE_FP32", "1").strip().lower() not in {"0", "false", "off"}:
            try:
                self.pipe.vae.to(dtype=torch.float32)
            except Exception:
                pass

        if offload_mode == "sequential":
            self.pipe.enable_sequential_cpu_offload(gpu_id=self.device_index)
            offload_label = "6 GB-safe sequential CPU offload"
        else:
            self.pipe.enable_model_cpu_offload(gpu_id=self.device_index)
            offload_label = "model CPU offload"

        if hasattr(self.pipe.vae, "enable_tiling"):
            self.pipe.vae.enable_tiling()
        if hasattr(self.pipe.vae, "enable_slicing"):
            self.pipe.vae.enable_slicing()

        self._clear_cuda_cache()
        self._report(
            callback,
            f"Wan2.1 ready ({backend_label}; {offload_label})",
            0.16,
        )

    @staticmethod
    def _quality_prompt(prompt: str) -> str:
        return (
            f"{prompt.strip()}. Premium family animation for children, stable character identity, "
            "same facial features and body proportions throughout the shot, consistent costume and colors, "
            "clean silhouette, controlled readable motion, coherent background geometry, polished soft lighting, "
            "clean anatomy, no text, watermark, flicker, jitter, morphing or character redesign."
        )

    def generate(self, prompt: str, seed: int | float = -1, callback: Progress = None) -> Path:
        if not prompt or not prompt.strip():
            raise ValueError("Describe the Wan2.1 clip first.")
        self._load(callback)
        actual_seed = self._seed(seed)
        enhanced_prompt = self._quality_prompt(prompt)

        self._report(
            callback,
            f"Wan2.1 {self.backend or 'auto'} rendering {WAN_FRAMES} native frames on GPU {self.device_index}",
            0.20,
        )
        self._clear_cuda_cache()
        try:
            with torch.cuda.device(self.device_index), torch.inference_mode():
                frames = self.pipe(
                    prompt=enhanced_prompt,
                    negative_prompt=(
                        "blurry, low detail, distorted, flicker, jitter, morphing, identity drift, character redesign, "
                        "costume change, color shift, duplicate character, extra limbs, broken anatomy, text, watermark"
                    ),
                    height=WAN_HEIGHT,
                    width=WAN_WIDTH,
                    num_frames=WAN_FRAMES,
                    num_inference_steps=WAN_STEPS,
                    guidance_scale=6.0,
                    generator=torch.Generator(device=f"cuda:{self.device_index}").manual_seed(actual_seed),
                ).frames[0]
        except torch.OutOfMemoryError as exc:
            self._clear_cuda_cache()
            raise RuntimeError(
                "Wan2.1 exhausted this GPU. For a 6 GB RTX 4050 use the GGUF path: "
                "python setup_wan.py --download-gguf, set WAN_BACKEND=gguf, and keep WAN_OFFLOAD_MODE=sequential."
            ) from exc

        from diffusers.utils import export_to_video

        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        native = OUTPUTS_DIR / f"wan_native_{stamp}.mp4"
        final = OUTPUTS_DIR / f"wan_youtube_1080p_{WAN_DELIVERY_FPS}fps_{stamp}.mp4"
        self._report(callback, "Encoding native Wan2.1 clip", 0.90)
        export_to_video(frames, str(native), fps=WAN_FPS)
        del frames
        self._clear_cuda_cache()

        def enhancement_status(message: str) -> None:
            self._report(callback, message, 0.96)

        self._report(callback, "Preparing smooth enhanced 1080p delivery", 0.93)
        enhance_wan_delivery(
            native,
            final,
            width=1920,
            height=1080,
            native_fps=WAN_FPS,
            target_fps=WAN_DELIVERY_FPS,
            gpu_id=self.device_index,
            callback=enhancement_status,
        )
        self._report(callback, f"Saved {final.name}", 1.0)
        return final

    def generate_story(
        self,
        story: str,
        target_seconds: int,
        seed: int | float = -1,
        callback: Progress = None,
    ) -> tuple[Path, tuple[str, ...]]:
        beats = build_story_sequence(story, target_seconds)
        clips: list[Path] = []
        context = " ".join(story.split())[:700]
        total = len(beats)
        base_seed = self._seed(seed)
        continuity_lock = (
            "Continuity lock: the exact same main character identity, face, age, body proportions, hairstyle/fur, "
            "costume, colors and art style must persist in every scene. Preserve the same world design and lighting "
            "language. Do not redesign the character between shots."
        )

        for index, beat in enumerate(beats):
            scene_prompt = (
                f"Story clip {index + 1} of {total}. Full story context: {context}. "
                f"Current event only: {beat}. {continuity_lock} "
                "Keep motion simple and readable: one main action and one controlled camera move."
            )

            def scene_callback(message: str, value: float, scene_index: int = index) -> None:
                if callback:
                    callback(f"Scene {scene_index + 1}/{total}: {message}", (scene_index + value) / total)

            clips.append(self.generate(scene_prompt, base_seed, scene_callback))

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        assembled = OUTPUTS_DIR / f"wan_story_assembled_{stamp}.mp4"
        final = OUTPUTS_DIR / f"wan_story_1080p_{WAN_DELIVERY_FPS}fps_{stamp}.mp4"
        self._report(callback, "Combining ordered enhanced Wan2.1 clips", 0.995)
        concatenate_videos_streaming(clips, assembled, target_fps=WAN_DELIVERY_FPS)
        export_delivery(
            assembled,
            final,
            1920,
            1080,
            enhance_quality=True,
            target_fps=WAN_DELIVERY_FPS,
            duration_seconds=target_seconds,
        )
        self._report(callback, f"Saved {final.name}", 1.0)
        return final, beats


GENERATOR = WanVideoGenerator(device_index=int(os.getenv("WAN_GPU_ID", "0")))
