"""LTX Video Director Studio configuration tuned for RTX 4050 / 16 GB RAM laptops."""
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
COMICS_NEGATIVE_PROMPT = (
    "photorealistic skin, live-action look, inconsistent line art, changing costume, changing face, "
    "warped anatomy, extra limbs, unreadable text, panel borders, watermark, flicker, jitter"
)
REAL_WORLD_NEGATIVE_PROMPT = (
    "cartoon, anime, illustration, plastic skin, waxy face, impossible physics, warped hands, "
    "duplicate people, unstable identity, flicker, jitter, oversaturated CGI, watermark"
)
ACTION_NEGATIVE_PROMPT = (
    "chaotic camera, impossible body motion, broken anatomy, teleporting, duplicated limbs, "
    "object morphing, flicker, jitter, motion smear, unreadable action, watermark"
)

CARTOON_STYLES = {
    "Premium 3D Kids Animation": "premium stylized 3D family animation, soft global illumination, expressive faces, polished materials, colorful cinematic lighting",
    "Claymation": "handmade claymation, tactile clay texture, miniature sets, charming stop-motion feel, soft studio lighting",
    "2D Storybook": "high-end 2D storybook animation, clean line art, painterly backgrounds, appealing shapes, rich color harmony",
    "Anime Adventure": "polished anime film style, expressive character acting, detailed painted backgrounds, dynamic cinematic composition",
    "Soft Ghibli-inspired": "gentle hand-painted fantasy animation, warm natural light, whimsical environment, soft expressive movement",
    "Comic Toon": "bold cartoon illustration, clean cel shading, energetic poses, colorful graphic shapes, playful cinematic motion",
}

COMICS_STYLES = {
    "Motion Comic": "premium motion-comic aesthetic, crisp inked contours, controlled parallax, selective animated details, cinematic panel composition",
    "Graphic Novel": "high-end graphic novel art, dramatic ink work, textured shadows, mature cinematic composition, rich controlled color grading",
    "Manga": "polished manga-inspired animation, confident black line art, screen-tone texture, expressive poses, dynamic speed accents used sparingly",
    "Western Superhero": "bold western superhero comic art, clean anatomy, strong silhouettes, saturated inks, dramatic rim lighting",
    "Cel-Shaded Comic": "3D cel-shaded comic animation, clean contour lines, graphic shadow shapes, vibrant controlled palette, cinematic depth",
    "Retro Pulp": "vintage pulp-comic illustration, halftone texture, limited print palette, dramatic composition, subtle paper grain",
}

REAL_WORLD_STYLES = {
    "Cinematic Live Action": "photorealistic live-action cinema, natural skin and materials, physically plausible motion, cinematic dynamic range",
    "Documentary": "observational documentary realism, natural available light, authentic textures, restrained camera language, believable human behavior",
    "Product Commercial": "premium commercial cinematography, precise product detail, controlled reflections, polished studio lighting, intentional camera motion",
    "Travel Film": "high-end travel cinematography, natural environmental detail, atmospheric depth, authentic motion, elegant camera movement",
    "Corporate / Presenter": "professional real-world presentation video, natural gestures, clean modern environment, flattering practical lighting, stable framing",
    "Lifestyle": "natural lifestyle advertising, believable candid behavior, soft daylight, realistic textures, clean cinematic composition",
}

ACTION_STYLES = {
    "Cinematic Action": "grounded cinematic action, readable choreography, realistic inertia, strong spatial continuity, polished film lighting",
    "Sports": "professional sports cinematography, athletic body mechanics, readable movement, realistic speed, broadcast-quality tracking",
    "Chase": "tense grounded chase sequence, clear pursuit geography, realistic acceleration and momentum, cinematic suspense",
    "Parkour": "athletic parkour action, precise body mechanics, believable jumps and landings, strong environmental interaction",
    "Martial Arts": "clean martial-arts choreography, readable strikes and defensive movement, balanced footwork, physically coherent contact",
    "Adventure": "cinematic adventure action, clear objective, believable environmental hazards, readable movement, epic but grounded staging",
}

SHOT_PRESETS = [
    "Wide establishing shot",
    "Medium shot",
    "Medium close-up",
    "Close-up",
    "Low-angle hero shot",
    "High-angle overview",
    "Over-the-shoulder shot",
    "Tracking side profile",
]

CAMERA_MOTIONS = [
    "Tripod-locked camera",
    "Gentle dolly in",
    "Slow dolly out",
    "Stable lateral tracking",
    "Slow cinematic pan",
    "Subtle handheld camera",
    "Controlled orbit around the subject",
    "Low-angle forward tracking",
]

LIGHTING_PRESETS = [
    "Soft natural daylight",
    "Golden-hour backlight",
    "Overcast diffused light",
    "Dramatic side light",
    "Neon night lighting",
    "High-key studio lighting",
    "Low-key cinematic lighting",
    "Warm practical interior lighting",
]

ACTION_INTENSITY = {
    "Controlled": "controlled energy, deliberate readable movement, stable pacing",
    "Dynamic": "dynamic energy, fast but readable movement, clear start and finish",
    "High Impact": "high-impact action with strong momentum, but preserve readable choreography and physical continuity",
}

STUDIO_EXAMPLES = {
    "comics": {
        "Neon Detective": {
            "subject": "A masked detective in a long charcoal coat",
            "action": "steps from a doorway, raises a glowing evidence card, then turns toward the camera",
            "environment": "a rain-soaked neon alley with reflected signs and drifting steam",
            "extra": "Keep the coat, mask shape and cyan evidence glow identical from first frame to last.",
        },
        "Superhero Rooftop": {
            "subject": "A red-and-gold comic-book hero with a short cape",
            "action": "lands on one knee, rises, and looks across the city skyline",
            "environment": "a high rooftop at sunset with distant skyscrapers",
            "extra": "Strong silhouette, clean cape motion, no panel cuts inside the shot.",
        },
    },
    "real_world": {
        "Coffee Commercial": {
            "subject": "A barista holding a ceramic cup of fresh coffee",
            "action": "sets the cup on a walnut counter while steam curls upward",
            "environment": "a premium modern cafe during warm early-morning light",
            "extra": "Preserve realistic ceramic, wood and skin textures with subtle condensation and natural hand motion.",
        },
        "Travel Walk": {
            "subject": "A traveler in a light linen shirt carrying a small backpack",
            "action": "walks through the market and briefly looks toward a fruit stall",
            "environment": "a lively old-city street market in soft morning daylight",
            "extra": "Background pedestrians move naturally without drawing focus from the subject.",
        },
    },
    "action": {
        "Parkour Vault": {
            "subject": "An athletic parkour runner in dark training clothes",
            "action": "runs three steps and performs one clean speed vault over a waist-high concrete barrier",
            "environment": "an open urban rooftop with clear landing space",
            "extra": "Show hand contact, leg clearance, landing impact and forward momentum clearly.",
        },
        "Cyclist Sprint": {
            "subject": "A road cyclist wearing a blue racing jersey and helmet",
            "action": "accelerates out of the saddle and sprints past the camera",
            "environment": "a closed mountain road with dry asphalt and distant hills",
            "extra": "Keep wheel rotation, pedaling cadence and bike geometry mechanically coherent.",
        },
    },
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
