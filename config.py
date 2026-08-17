"""LTX-2.3 Video Studio configuration tuned for a 16 GB NVIDIA GPU."""

from pathlib import Path
import os

PROJECT_ROOT = Path(__file__).parent.resolve()
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
STATIC_DIR = PROJECT_ROOT / "static"
LTX_SOURCE_DIR = PROJECT_ROOT / "vendor" / "LTX-2"

MODELS_DIR.mkdir(exist_ok=True)
OUTPUTS_DIR.mkdir(exist_ok=True)
STATIC_DIR.mkdir(exist_ok=True)

HF_REPO_ID = "Lightricks/LTX-2.3"
MODEL_FILES = {
    "checkpoint": {
        "filename": "ltx-2.3-22b-distilled-1.1.safetensors",
        "description": "LTX-2.3 distilled 22B checkpoint",
    },
    "spatial_upscaler": {
        "filename": "ltx-2.3-spatial-upscaler-x2-1.1.safetensors",
        "description": "LTX-2.3 2x latent spatial upscaler",
    },
}

# Official text encoder listed by the LTX-2.3 model documentation.
GEMMA_REPO_ID = "google/gemma-3-12b-it-qat-q4_0-unquantized"
GEMMA_DIR = MODELS_DIR / "gemma-3-12b-it-qat-q4_0-unquantized"

RESOLUTION_PRESETS = {
    "Fast (384×256)": {"width": 384, "height": 256},
    "Balanced (512×320) — Recommended 16GB": {"width": 512, "height": 320},
    "Quality (640×384)": {"width": 640, "height": 384},
    "HD (768×512) — Slower": {"width": 768, "height": 512},
}
DEFAULT_RESOLUTION = "Balanced (512×320) — Recommended 16GB"

DURATION_PRESETS = {
    "2 seconds (49 frames)": 49,
    "4 seconds (97 frames) — Recommended": 97,
    "5 seconds (121 frames)": 121,
    "7 seconds (161 frames)": 161,
    "8 seconds (193 frames)": 193,
    "10 seconds (241 frames)": 241,
}
DEFAULT_DURATION = "4 seconds (97 frames) — Recommended"

DEFAULT_NUM_INFERENCE_STEPS = 8
DEFAULT_GUIDANCE_SCALE = 3.0
DEFAULT_FPS = 24
DEFAULT_SEED = -1
MAX_CONTINUATION_CLIPS = 3
CROSSFADE_DURATION = 0.5

# 16 GB profile. The distilled pipeline has a fixed fast schedule.
TARGET_VRAM_GB = 16
ENABLE_FP8 = True
OFFLOAD_MODE = "auto"  # >=36 GB RAM -> CPU cache, otherwise disk streaming.
KEEP_MODEL_LOADED = True
IMAGE_CONDITIONING_STRENGTH = 1.0

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

PROMPT_PRESETS = {
    "🐰 Cartoon Rhyme": "A cute animated cartoon rabbit hopping joyfully through a sunlit meadow full of colorful flowers, vibrant 3D animation style, children's cartoon, bright pastel colors, smooth motion, whimsical and playful atmosphere",
    "🌆 Cinematic City": "A sweeping cinematic aerial shot of a futuristic city at golden hour, towering glass skyscrapers reflecting warm sunlight, flying vehicles weaving between buildings, volumetric fog, movie quality",
    "🌊 Ocean Waves": "A breathtaking slow-motion shot of turquoise ocean waves crashing on a pristine white sandy beach, crystal clear water, golden sunset light, photorealistic",
    "🚀 Sci-Fi Space": "A massive spaceship emerging from hyperspace near a ringed planet, stars streaking in the background, detailed hull with glowing engines, cinematic lighting",
    "🎨 Abstract Art": "An abstract fluid art animation, vibrant swirling colors, organic flowing movements, mesmerizing patterns, smooth transitions",
    "🌸 Nature Close-up": "An extreme macro close-up of a dewdrop on a rose petal, morning sunlight creating rainbow refractions, shallow depth of field, photorealistic",
    "🏰 Fantasy World": "A magical fantasy castle perched on a floating island in the sky, waterfalls cascading into clouds below, dragons soaring around the towers, golden hour lighting",
    "🎭 Realistic Portrait": "A photorealistic portrait of a person standing in falling autumn leaves, warm golden backlight, shallow depth of field, gentle wind blowing hair",
}
