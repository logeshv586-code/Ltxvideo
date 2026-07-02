"""
LTX-2.3 Video Generation Platform — Configuration
Model paths, resolution presets, and default generation parameters.
"""

import os
from pathlib import Path

# ──────────────────────────────────────────────
# Project Paths
# ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.resolve()
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
STATIC_DIR = PROJECT_ROOT / "static"

# Create directories
MODELS_DIR.mkdir(exist_ok=True)
OUTPUTS_DIR.mkdir(exist_ok=True)
STATIC_DIR.mkdir(exist_ok=True)

# ──────────────────────────────────────────────
# HuggingFace Model Repository
# ──────────────────────────────────────────────
HF_REPO_ID = "Lightricks/LTX-2.3"

MODEL_FILES = {
    "checkpoint": {
        "filename": "ltx-2.3-22b-distilled-1.1.safetensors",
        "description": "Distilled model checkpoint (22B params, optimized for 8-step inference)",
    },
    "spatial_upscaler": {
        "filename": "ltx-2.3-spatial-upscaler-x2-1.1.safetensors",
        "description": "2x spatial resolution upscaler",
    },
    "distilled_lora": {
        "filename": "ltx-2.3-22b-distilled-lora-384-1.1.safetensors",
        "description": "Distilled LoRA for Stage 2 refinement",
    },
}

GEMMA_REPO_ID = "google/gemma-3-4b-it"
GEMMA_DIR = MODELS_DIR / "gemma-3-4b-it"

# ──────────────────────────────────────────────
# Resolution Presets (safe for 6GB VRAM)
# Width and height must be divisible by 32
# ──────────────────────────────────────────────
RESOLUTION_PRESETS = {
    "Low (384×256) — Fastest": {"width": 384, "height": 256},
    "Medium (512×320) — Balanced": {"width": 512, "height": 320},
    "High (640×384) — Better Quality": {"width": 640, "height": 384},
    "HD (768×512) — Slow, High Quality": {"width": 768, "height": 512},
}

DEFAULT_RESOLUTION = "Medium (512×320) — Balanced"

# ──────────────────────────────────────────────
# Frame Count Presets
# Frame count must follow 8k+1 pattern (e.g., 97, 121, 161, 193, 241)
# At 24fps: 97 frames ≈ 4s, 121 ≈ 5s, 161 ≈ 6.7s, 193 ≈ 8s, 241 ≈ 10s
# ──────────────────────────────────────────────
DURATION_PRESETS = {
    "2 seconds (49 frames)": 49,      # 8*6+1
    "4 seconds (97 frames)": 97,      # 8*12+1
    "5 seconds (121 frames)": 121,    # 8*15+1
    "7 seconds (161 frames)": 161,    # 8*20+1
    "8 seconds (193 frames)": 193,    # 8*24+1
    "10 seconds (241 frames)": 241,   # 8*30+1
}

DEFAULT_DURATION = "5 seconds (121 frames)"

# ──────────────────────────────────────────────
# Generation Defaults
# ──────────────────────────────────────────────
DEFAULT_NUM_INFERENCE_STEPS = 8       # Distilled model needs fewer steps
DEFAULT_GUIDANCE_SCALE = 3.0
DEFAULT_FPS = 24
DEFAULT_SEED = -1                     # -1 = random
MAX_CONTINUATION_CLIPS = 3           # Up to 3 clips for 30s total
CROSSFADE_DURATION = 0.5             # Seconds of crossfade between clips

# ──────────────────────────────────────────────
# Memory Optimization (for 6GB VRAM)
# ──────────────────────────────────────────────
ENABLE_CPU_OFFLOAD = True
ENABLE_VAE_SLICING = True
ENABLE_VAE_TILING = True
ENABLE_ATTENTION_SLICING = True
TORCH_DTYPE = "bfloat16"             # or "float16" for older GPUs

# ──────────────────────────────────────────────
# Environment Variables (set at startup)
# ──────────────────────────────────────────────
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ──────────────────────────────────────────────
# Prompt Presets
# ──────────────────────────────────────────────
PROMPT_PRESETS = {
    "🐰 Cartoon Rhyme": "A cute animated cartoon rabbit hopping joyfully through a sunlit meadow full of colorful flowers, vibrant 3D animation style, children's cartoon, bright pastel colors, smooth motion, whimsical and playful atmosphere",
    "🌆 Cinematic City": "A sweeping cinematic aerial shot of a futuristic city at golden hour, towering glass skyscrapers reflecting warm sunlight, flying vehicles weaving between buildings, volumetric fog, 4K, ultra detailed, movie quality",
    "🌊 Ocean Waves": "A breathtaking slow-motion shot of turquoise ocean waves crashing on a pristine white sandy beach, crystal clear water, golden sunset light, photorealistic, 4K, National Geographic quality",
    "🚀 Sci-Fi Space": "A massive spaceship emerging from hyperspace near a ringed planet, stars streaking in the background, detailed hull with glowing engines, cinematic lighting, epic science fiction scene, 4K",
    "🎨 Abstract Art": "An abstract fluid art animation, vibrant swirling colors of deep purple, electric blue, and molten gold, organic flowing movements, mesmerizing patterns, smooth transitions, artistic and hypnotic",
    "🌸 Nature Close-up": "An extreme macro close-up of a dewdrop on a rose petal, morning sunlight creating rainbow refractions, shallow depth of field, photorealistic, 8K detail, time-lapse of the dewdrop slowly sliding",
    "🏰 Fantasy World": "A magical fantasy castle perched on a floating island in the sky, waterfalls cascading off the edges into clouds below, dragons soaring around the towers, golden hour lighting, Studio Ghibli inspired",
    "🎭 Realistic Portrait": "A photorealistic portrait of a person standing in falling autumn leaves, warm golden backlight, shallow depth of field, cinematic color grading, gentle wind blowing hair, 4K ultra detailed",
}
