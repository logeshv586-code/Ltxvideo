"""LTX Cartoon Studio configuration tuned for RTX 4050 / 16 GB RAM laptops."""
from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.resolve()
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
STATIC_DIR = PROJECT_ROOT / "static"
OFFLOAD_DIR = MODELS_DIR / "offload"
HF_CACHE_DIR = MODELS_DIR / "hf-cache"
for directory in (MODELS_DIR, OUTPUTS_DIR, STATIC_DIR, OFFLOAD_DIR, HF_CACHE_DIR):
    directory.mkdir(parents=True, exist_ok=True)

HF_REPO_ID = "Lightricks/LTX-Video"
MODEL_VARIANT = "LTX-Video 0.9.8 / 2B low-memory Diffusers path"
DEFAULT_FPS = 30
GPU_MEMORY_BUDGET = "5GiB"
CPU_MEMORY_BUDGET = "8GiB"
ENABLE_8BIT = True
ENABLE_VAE_TILING = True

RESOLUTION_PRESETS = {
    "4050 Fast • 384×224 • 16:9": {"width": 384, "height": 224, "risk": "low"},
    "4050 Balanced • 512×288 • 16:9": {"width": 512, "height": 288, "risk": "medium"},
    "4050 Portrait • 288×512 • 9:16": {"width": 288, "height": 512, "risk": "medium"},
    "4050 Square • 384×384 • 1:1": {"width": 384, "height": 384, "risk": "medium"},
    "4050 Cinema • 512×224": {"width": 512, "height": 224, "risk": "medium"},
}
DEFAULT_RESOLUTION = "4050 Fast • 384×224 • 16:9"

DURATION_PRESETS = {
    "1.6 sec • 49 frames • safest": 49,
    "3.2 sec • 97 frames • fast": 97,
    "4.0 sec • 121 frames • recommended": 121,
    "5.4 sec • 161 frames • heavier": 161,
    "6.4 sec • 193 frames • heavy": 193,
    "8.0 sec • 241 frames • max practical": 241,
}
DEFAULT_DURATION = "4.0 sec • 121 frames • recommended"
MAX_NATIVE_FRAMES = 241
MAX_STORY_SCENES = 24
DEFAULT_STORY_SCENES = 4
DEFAULT_NUM_INFERENCE_STEPS = 20
DEFAULT_GUIDANCE_SCALE = 3.0
DEFAULT_SEED = -1

NEGATIVE_PROMPT = (
    "worst quality, inconsistent motion, jitter, flicker, blurry, deformed, "
    "duplicate character, changing costume, changing face, watermark, text artifacts"
)

CARTOON_STYLES = {
    "Premium 3D Kids Animation": "premium stylized 3D family animation, soft global illumination, expressive faces, polished materials, colorful cinematic lighting",
    "Claymation": "handmade claymation, tactile clay texture, miniature sets, charming stop-motion feel, soft studio lighting",
    "2D Storybook": "high-end 2D storybook animation, clean line art, painterly backgrounds, appealing shapes, rich color harmony",
    "Anime Adventure": "polished anime film style, expressive character acting, detailed painted backgrounds, dynamic cinematic composition",
    "Soft Ghibli-inspired": "gentle hand-painted fantasy animation, warm natural light, whimsical environment, soft expressive movement",
    "Comic Toon": "bold cartoon illustration, clean cel shading, energetic poses, colorful graphic shapes, playful cinematic motion",
}

CAMERA_PRESETS = [
    "gentle dolly in",
    "slow cinematic pan",
    "stable medium tracking shot",
    "wide establishing shot with subtle parallax",
    "character close-up with soft push-in",
]

EXPORT_PRESETS = {
    "Native MP4": None,
    "720p Landscape": (1280, 720),
    "1080p Landscape": (1920, 1080),
    "720p Portrait": (720, 1280),
    "1080p Portrait": (1080, 1920),
    "1080p Square": (1080, 1080),
}

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("HF_HOME", str(HF_CACHE_DIR))
