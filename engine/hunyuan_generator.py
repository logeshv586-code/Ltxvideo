"""HunyuanVideo-1.5 backend for RTX 4080-class GPUs.

This adapter intentionally invokes Tencent's official HunyuanVideo-1.5 source
instead of duplicating model internals in this repository.  The recommended
path is 480p image-to-video step distillation (8/12 steps) with CPU/group
offloading and the official super-resolution pipeline.
"""
from __future__ import annotations

import os
import random
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

from PIL import Image

from config import MODELS_DIR, OUTPUTS_DIR, PROJECT_ROOT

Progress = Callable[[str, float], None] | None

HUNYUAN_SOURCE_DIR = Path(os.getenv("HUNYUAN_SOURCE_DIR", PROJECT_ROOT / "third_party" / "HunyuanVideo-1.5"))
HUNYUAN_MODEL_DIR = Path(os.getenv("HUNYUAN_MODEL_DIR", MODELS_DIR / "hunyuanvideo-1.5"))


@dataclass(frozen=True)
class HunyuanPreset:
    name: str
    steps: int
    enable_sr: bool
    description: str


HUNYUAN_PRESETS = {
    "Final • 12 steps • SR": HunyuanPreset(
        name="Final • 12 steps • SR",
        steps=12,
        enable_sr=True,
        description="Recommended Moon Cookie final render: best speed/quality balance with official SR.",
    ),
    "Fast • 8 steps • SR": HunyuanPreset(
        name="Fast • 8 steps • SR",
        steps=8,
        enable_sr=True,
        description="Faster review render with comparable step-distilled quality and official SR.",
    ),
    "Draft • 4 steps • 480p": HunyuanPreset(
        name="Draft • 4 steps • 480p",
        steps=4,
        enable_sr=False,
        description="Rapid storyboard preview. Faster but visibly lower quality.",
    ),
}

ASPECT_RATIOS = ("16:9", "9:16", "1:1")
DEFAULT_HUNYUAN_PRESET = "Final • 12 steps • SR"
DEFAULT_VIDEO_FRAMES = 121
DEFAULT_FPS = 24


class HunyuanVideoGenerator:
    """Run the official HunyuanVideo-1.5 CLI with safe RTX 4080 defaults."""

    def __init__(self, source_dir: Path | None = None, model_dir: Path | None = None) -> None:
        self.source_dir = Path(source_dir or HUNYUAN_SOURCE_DIR)
        self.model_dir = Path(model_dir or HUNYUAN_MODEL_DIR)

    @staticmethod
    def _report(callback: Progress, message: str, value: float) -> None:
        print(f"[{value * 100:5.1f}%] {message}")
        if callback:
            callback(message, value)

    @staticmethod
    def _seed(seed: int) -> int:
        return random.randint(0, 2**31 - 1) if seed is None or int(seed) < 0 else int(seed)

    @staticmethod
    def _output_path(prefix: str = "hunyuan_i2v") -> Path:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        return OUTPUTS_DIR / f"{prefix}_{stamp}.mp4"

    def readiness(self) -> tuple[bool, str]:
        generate_py = self.source_dir / "generate.py"
        if not generate_py.exists():
            return False, (
                f"Hunyuan source is not installed at {self.source_dir}. "
                "Run `python setup_hunyuan.py --install-code`."
            )
        if not self.model_dir.exists():
            return False, (
                f"Hunyuan checkpoints are not installed at {self.model_dir}. "
                "Run `python setup_hunyuan.py --show-downloads`."
            )
        required = [
            self.model_dir / "transformer" / "480p_i2v_step_distilled",
            self.model_dir / "vae",
            self.model_dir / "text_encoder",
            self.model_dir / "vision_encoder",
        ]
        missing = [str(path.relative_to(self.model_dir)) for path in required if not path.exists()]
        if missing:
            return False, "Checkpoint folder is incomplete; missing: " + ", ".join(missing)
        return True, "HunyuanVideo-1.5 480p I2V step-distilled backend is ready."

    def build_command(
        self,
        prompt: str,
        image_path: Path,
        output_path: Path,
        negative_prompt: str = "",
        aspect_ratio: str = "16:9",
        steps: int = 12,
        seed: int = 123,
        enable_sr: bool = True,
        overlap_group_offloading: bool = False,
        save_pre_sr_video: bool = False,
    ) -> list[str]:
        if aspect_ratio not in ASPECT_RATIOS:
            raise ValueError(f"Unsupported aspect ratio: {aspect_ratio}")
        if int(steps) not in (4, 8, 12):
            raise ValueError("Step-distilled Hunyuan I2V supports 4, 8 or 12 steps in this studio.")

        # Official generate.py supports one-GPU execution through torchrun. Using
        # `python -m torch.distributed.run` also works reliably on Windows where
        # the torchrun executable may not be on PATH.
        return [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--nproc_per_node=1",
            str(self.source_dir / "generate.py"),
            "--prompt",
            prompt,
            "--negative_prompt",
            negative_prompt or "",
            "--image_path",
            str(image_path),
            "--resolution",
            "480p",
            "--aspect_ratio",
            aspect_ratio,
            "--seed",
            str(int(seed)),
            "--num_inference_steps",
            str(int(steps)),
            "--video_length",
            str(DEFAULT_VIDEO_FRAMES),
            "--rewrite",
            "false",
            "--cfg_distilled",
            "false",
            "--enable_step_distill",
            "true",
            "--sparse_attn",
            "false",
            "--use_sageattn",
            "false",
            # Step distillation and cache are deliberately never enabled together.
            "--enable_cache",
            "false",
            "--offloading",
            "true",
            "--group_offloading",
            "true",
            "--overlap_group_offloading",
            "true" if overlap_group_offloading else "false",
            "--dtype",
            "bf16",
            "--sr",
            "true" if enable_sr else "false",
            "--save_pre_sr_video",
            "true" if save_pre_sr_video else "false",
            "--output_path",
            str(output_path),
            "--model_path",
            str(self.model_dir),
        ]

    def generate_image_to_video(
        self,
        prompt: str,
        image: Image.Image,
        negative_prompt: str = "",
        aspect_ratio: str = "16:9",
        preset: str = DEFAULT_HUNYUAN_PRESET,
        seed: int = -1,
        overlap_group_offloading: bool = False,
        save_pre_sr_video: bool = False,
        progress_callback: Progress = None,
    ) -> Path:
        ready, status = self.readiness()
        if not ready:
            raise RuntimeError(status)
        if image is None:
            raise ValueError("A reference image is required for Hunyuan 480p step-distilled I2V.")
        if not prompt or not prompt.strip():
            raise ValueError("Describe the movement/action for this 5-second shot.")

        selected = HUNYUAN_PRESETS.get(preset)
        if selected is None:
            raise ValueError(f"Unknown Hunyuan preset: {preset}")

        actual_seed = self._seed(seed)
        output = self._output_path()
        self._report(progress_callback, f"Preparing Hunyuan shot · {selected.steps} steps · seed {actual_seed}", 0.03)

        with tempfile.TemporaryDirectory(prefix="ltx_hunyuan_") as temp_dir:
            reference_path = Path(temp_dir) / "reference.png"
            image.convert("RGB").save(reference_path, format="PNG")
            command = self.build_command(
                prompt=prompt.strip(),
                image_path=reference_path,
                output_path=output,
                negative_prompt=negative_prompt,
                aspect_ratio=aspect_ratio,
                steps=selected.steps,
                seed=actual_seed,
                enable_sr=selected.enable_sr,
                overlap_group_offloading=overlap_group_offloading,
                save_pre_sr_video=save_pre_sr_video,
            )

            env = os.environ.copy()
            env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:128")
            env.setdefault("TOKENIZERS_PARALLELISM", "false")
            self._report(progress_callback, "Launching official HunyuanVideo-1.5 pipeline…", 0.08)

            process = subprocess.Popen(
                command,
                cwd=str(self.source_dir),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
            recent: list[str] = []
            assert process.stdout is not None
            for line in process.stdout:
                line = line.rstrip()
                if not line:
                    continue
                print(f"[Hunyuan] {line}")
                recent.append(line)
                recent = recent[-30:]
                # The official CLI does not expose stable machine-readable progress,
                # so keep the UI responsive with a conservative in-progress marker.
                self._report(progress_callback, line[-180:], 0.50)

            return_code = process.wait()
            if return_code != 0:
                tail = "\n".join(recent[-12:])
                raise RuntimeError(f"Hunyuan generation failed with exit code {return_code}.\n{tail}")

        if not output.exists() or output.stat().st_size == 0:
            raise RuntimeError("Hunyuan completed without producing the expected MP4 file.")

        self._report(progress_callback, f"Saved {output.name}", 1.0)
        return output


def model_storage_summary() -> str:
    """Return a short installation status for the UI."""
    generator = HunyuanVideoGenerator()
    ready, message = generator.readiness()
    state = "READY" if ready else "SETUP REQUIRED"
    return f"**Hunyuan backend: {state}**  \n{message}  \nModel path: `{generator.model_dir}`"
