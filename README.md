<p align="center">
  <img src="docs/assets/hero.svg" alt="LTX Video Director Studio" width="100%" />
</p>

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white" />
  <img alt="GPU" src="https://img.shields.io/badge/Target-RTX%204050-76B900?logo=nvidia&logoColor=white" />
  <img alt="Model" src="https://img.shields.io/badge/LTX--Video-2B%20Low%20Memory-8B5CF6" />
  <img alt="Offline" src="https://img.shields.io/badge/Generation-Local%20%2F%20Offline-06B6D4" />
  <img alt="Modes" src="https://img.shields.io/badge/Studios-General%20%7C%20Comics%20%7C%20Real%20World%20%7C%20Action%20%7C%20Cartoon-EC4899" />
</p>

# LTX Video Director Studio

A premium local AI video workspace built around **Lightricks LTX-Video**, tuned for a laptop-class **RTX 4050 (6 GB VRAM) + 16 GB RAM** setup. Instead of forcing every job through one generic prompt box, the UI now separates creative work into dedicated studios with mode-aware prompt direction.

> **No cloud video-generation API is required after the first model download.** Hugging Face is used to fetch and cache the official open model files on first setup.

## Creative Studios

| Studio | Best for | What the UI adds |
|---|---|---|
| ✨ **General Studio** | Free-form T2V / I2V | Raw prompt + optional image reference |
| 📚 **Comics Studio** | Motion comics, manga, graphic novels, cel shading | Art-style lock, line/silhouette continuity, comic-specific negatives |
| 🌍 **Real-World Studio** | Live action, documentary, products, travel, presenters | Realism profile, materials, anatomy/physics anchors, natural camera language |
| ⚡ **Action Studio** | Sports, parkour, chases, martial arts, adventure | Action intensity, choreography, contact/trajectory notes, short-clip complexity control |
| 🧸 **Cartoon Story Studio** | Long multi-scene animated stories | Character bible, last-frame continuation, style continuity, scene stitching |

Every directed studio supports:

- **Prompt + optional reference image**
- Subject / character / product description
- One primary action beat
- Environment / location
- Style preset
- Shot / framing
- Camera movement
- Lighting
- Continuity / material / choreography notes
- Optional dialogue / ambience direction
- Mode-specific negative prompt
- Native RTX-4050 resolution and duration presets
- Seed, guidance and inference-step controls
- Native / 720p / 1080p landscape, portrait and square delivery exports
- **Director Prompt Preview** so you can see exactly what is sent to LTX before generation
- Quick starting examples for each studio

## Prompt-director workflow

LTX works best when a prompt reads like a **chronological shot description**, not a pile of tags. The directed studios therefore assemble the final model prompt in this order:

```text
Reference anchor (when image supplied)
        ↓
Subject + observable action
        ↓
Environment / spatial context
        ↓
Mode-specific visual direction
        ↓
Shot + one camera move
        ↓
Lighting / materials / continuity
        ↓
Optional dialogue / ambience
        ↓
Duration-aware motion constraint
        ↓
Stability / identity / physics anchors
```

For short clips, the Action and Real-World studios intentionally prefer **one main action beat + one camera move**. When an image is uploaded, the prompt builder treats it as the visual anchor and focuses on the motion/camera changes instead of redundantly describing the entire still image.

## Why this build is different

The original version of this repository pointed at **LTX-2.3 22B** and claimed 6 GB VRAM support. That was not a safe hardware match. The current architecture uses the official **2B LTX-Video Diffusers path**, loads the transformer and T5 encoder in **8-bit**, balances memory between GPU and CPU, enables VAE tiling/slicing, and avoids holding a complete long video in RAM.

<p align="center">
  <img src="docs/assets/hardware.svg" alt="RTX 4050 memory strategy" width="100%" />
</p>

## How many seconds can it generate?

For this RTX 4050 / 16 GB RAM profile, the UI exposes native clips from **49 to 241 frames**. At 30 FPS that is roughly **1.6 to 8.0 seconds per generated shot**. The default is **121 frames ≈ 4.0 seconds**.

For longer content, build multiple shots/scenes and stitch them. Cartoon Story Studio automates that process for animation.

| Goal | Resolution | Frames | Approx duration | Notes |
|---|---:|---:|---:|---|
| First GPU test | 384×224 | 49 | 1.6 s | Safest smoke test |
| Recommended scene | 384×224 | 121 | 4.0 s | **Default** |
| Better framing | 512×288 | 121 | 4.0 s | More VRAM pressure |
| Longer shot | 384×224 | 193 | 6.4 s | Heavy |
| Practical UI maximum | 384×224 | 241 | 8.0 s | Highest exposed native option |

## Comics Studio

Use Comics Studio when the final video should preserve a drawn visual language instead of drifting toward photorealism.

**Included presets:** Motion Comic, Graphic Novel, Manga, Western Superhero, Cel-Shaded Comic and Retro Pulp.

Recommended workflow:

1. Upload a reference image if the hero design must remain exact.
2. Describe the character and **one observable action**.
3. Choose the comic style and shot.
4. Keep camera movement controlled; motion-comic shots often benefit from subtle parallax rather than extreme movement.
5. Add costume / prop / palette continuity notes.
6. Preview the generated director prompt.
7. Generate.

## Real-World Studio

Use this section for believable live-action scenes, advertisements and documentary-style footage.

**Included presets:** Cinematic Live Action, Documentary, Product Commercial, Travel Film, Corporate / Presenter and Lifestyle.

The generated prompt automatically reinforces:

- natural anatomy and identity
- physically plausible motion
- believable weight / inertia
- realistic materials and surfaces
- coherent background geometry
- stable camera behavior

For Image-to-Video, describe **what should happen next** rather than re-describing the still image.

## Action Studio

Action Studio is designed for motion that must remain readable instead of turning into random fast movement.

**Included presets:** Cinematic Action, Sports, Chase, Parkour, Martial Arts and Adventure.

It adds:

- Controlled / Dynamic / High Impact intensity
- action timing rules based on clip duration
- choreography/contact notes
- momentum and trajectory consistency
- screen-direction and spatial continuity
- negatives for teleporting, broken anatomy, duplicated limbs and chaotic camera movement

For a 4-second clip, use **one stunt / one action beat**. Split multi-event action sequences into separate shots.

## Cartoon Story Studio

<p align="center">
  <img src="docs/assets/workflow.svg" alt="Cartoon story continuity workflow" width="100%" />
</p>

The story workflow is built to reduce the most common AI-cartoon failure: characters changing from scene to scene.

1. Write the full story or add **one scene beat per line**.
2. Define a **Character Bible**: face, colors, clothing, props, proportions and identifying details.
3. Optionally upload a character/style reference image for Scene 1.
4. Choose a cartoon style, scene count, native resolution and duration per scene.
5. Scene 1 is generated from text or your reference image.
6. The final frame is extracted and becomes the reference image for Scene 2.
7. The same character bible and visual style are injected into every scene prompt.
8. The process repeats through the story.
9. FFmpeg joins the clips using a streaming workflow so a long video does not fill 16 GB RAM with decoded frames.
10. Export the final movie as native MP4 or a delivery-size 720p/1080p landscape, portrait or square video.

## One-command start

### Windows

```powershell
python -m venv venv
venv\Scripts\activate
python run.py
```

### Linux

```bash
python -m venv venv
source venv/bin/activate
python run.py
```

`run.py` checks runtime packages, downloads the model cache if it is not already present, and launches the local UI at:

```text
http://127.0.0.1:7860
```

The first setup needs internet access for Hugging Face downloads. Later launches reuse the local cache.

## Download sizes

Native generation stays intentionally small for the RTX 4050. The export layer can produce:

- Native MP4
- 1280×720
- 1920×1080
- 720×1280
- 1080×1920
- 1080×1080

The 720p/1080p options are high-quality **delivery resizes using FFmpeg**. They do not create new AI detail. Native 1080p diffusion is intentionally disabled in this profile to reduce CUDA OOM risk.

## Memory architecture

```text
Studio form / raw prompt / reference image
                │
                ▼
Mode-aware prompt director (Comics / Real World / Action)
                │
                ▼
8-bit T5 Text Encoder ───────┐
                             │ GPU/CPU balanced placement
8-bit LTX 2B Transformer ────┤ RTX 4050 budget ≈ 5 GiB
                             │ CPU budget ≈ 8 GiB
Tiled/Sliced Video VAE ──────┘
                │
                ▼
Short Native MP4
                │
                ├── single shot → delivery export
                │
                └── cartoon story → last frame → next scene → FFmpeg join
```

## Repository layout

```text
Ltxvideo/
├── app.py                       # Multi-studio premium Gradio UI
├── run.py                       # One-command bootstrap + launch
├── config.py                    # RTX 4050 + creative-mode presets
├── download_models.py           # Hugging Face offline cache preparation
├── requirements.txt
├── engine/
│   ├── generator.py             # Quantized T2V / I2V backend
│   ├── prompt_builders.py       # Comics / Real World / Action prompt director
│   ├── storyboard.py            # Cartoon character + scene continuity
│   ├── memory_manager.py        # Hardware detection
│   └── video_processor.py       # Streaming join + delivery exports
├── static/
│   └── style.css                # Premium responsive UI
├── docs/assets/                 # README artwork
├── tests/
└── outputs/
```

## Testing

Run the lightweight repository tests without loading the video model:

```bash
python -m unittest discover -s tests -v
```

Syntax check:

```bash
python -m py_compile app.py config.py run.py download_models.py engine/*.py tests/*.py
```

The test suite covers:

- resolution alignment
- valid `8k+1` frame counts and RTX-4050 frame cap
- creative-mode preset presence
- comic style / identity lock behavior
- real-world physical realism anchors
- action short-clip complexity limits
- reference-image prompting behavior
- prompt-length safety
- cartoon storyboard continuity

A full video-generation test still requires an NVIDIA CUDA GPU and downloaded model weights.

## Troubleshooting

**CUDA out of memory**  
Close GPU-heavy applications, use `384×224`, choose 49/97/121 frames, and restart the app to clear VRAM.

**Windows becomes slow while loading**  
Close browsers/games and ensure the Windows page file is enabled. The text encoder and transformer are quantized, but 16 GB RAM is still a tight environment for modern video diffusion.

**The action looks chaotic**  
Reduce the prompt to one primary action beat and one camera move. Generate the next beat as a separate shot.

**The reference-image animation changes the character too much**  
Describe only the movement/camera change, keep a stable seed, use the conservative native resolution, and add exact identity/costume notes in the continuity field.

**1080p looks similar to native**  
That preset is a delivery resize, not an AI super-resolution pass.

## Model and upstream projects

- [Lightricks/LTX-Video](https://github.com/Lightricks/LTX-Video)
- [Lightricks LTX-Video on Hugging Face](https://huggingface.co/Lightricks/LTX-Video)
- [Hugging Face Diffusers LTX documentation](https://huggingface.co/docs/diffusers/api/pipelines/ltx_video)

## License

This repository contains application code around upstream model/runtime projects. Review the LTX-Video model license and all third-party licenses before commercial distribution.
