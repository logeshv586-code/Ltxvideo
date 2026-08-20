<p align="center">
  <img src="docs/assets/hero.svg" alt="LTX Video Director Studio" width="100%" />
</p>

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white" />
  <img alt="GPU" src="https://img.shields.io/badge/Target-RTX%204050-76B900?logo=nvidia&logoColor=white" />
  <img alt="Local" src="https://img.shields.io/badge/Generation-Local%20%2F%20Offline-06B6D4" />
  <img alt="License" src="https://img.shields.io/github/license/logeshv586-code/Ltxvideo" />
  <img alt="Stars" src="https://img.shields.io/github/stars/logeshv586-code/Ltxvideo?style=flat" />
  <img alt="Issues" src="https://img.shields.io/github/issues/logeshv586-code/Ltxvideo" />
</p>

# LTX Video Director Studio

**Local AI video generation for laptop-class NVIDIA GPUs.** Ltxvideo wraps Lightricks LTX-Video in a multi-studio Gradio workspace tuned around an **RTX 4050 6 GB VRAM + 16 GB RAM** target, with dedicated flows for text-to-video, image-to-video, action shots, comics, realistic scenes, and multi-scene cartoon continuity.

> **No paid cloud video-generation API is required after the first model download.** Model files are downloaded once and reused from the local cache.

## What you can do

| Workflow | What Ltxvideo adds |
|---|---|
| **Text → Video** | Directed chronological prompts, camera/shot controls, low-memory presets |
| **Image → Video** | Reference anchoring that focuses the prompt on motion and camera change |
| **Action Studio** | Action intensity, choreography, trajectory and camera-stability constraints |
| **Comics Studio** | Art-style, silhouette, line and costume continuity controls |
| **Real-World Studio** | Physical realism, anatomy, materials and stable-camera guidance |
| **Cartoon Story Studio** | Character bible, last-frame continuation, repeated style locking and FFmpeg scene stitching |
| **Delivery exports** | Native MP4 plus 720p/1080p landscape, portrait and square delivery sizes |

### Real RTX 4050 demo

A reproducible hardware demo is the next public-proof milestone. It is being tracked in **[Issue #7: Capture and publish the first real RTX 4050 demo reel](https://github.com/logeshv586-code/Ltxvideo/issues/7)**.

The project intentionally does **not** label cloud-generated footage as local RTX 4050 proof. The published demo should include the GPU name, prompt, seed, frame count, resolution and approximate generation time.

## Quick start

### Windows PowerShell

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

`run.py` checks the required runtime packages, prepares the local model cache when needed, and opens the UI at:

```text
http://127.0.0.1:7860
```

The first model setup requires internet access. Later launches reuse the local cache.

## Windows RTX 4050 first-run checklist

For the project's RTX 4050 / 16 GB RAM target, use this sequence for a first local launch.

### 1. Check the NVIDIA driver

Open PowerShell and run:

~~~powershell
nvidia-smi
~~~

Confirm that Windows can see your NVIDIA GPU before starting the application.

### 2. Create and activate the Python environment

Use Python 3.10+:

~~~powershell
python --version
python -m venv venv
venv\Scripts\activate
python -m pip install -r requirements.txt
~~~

### 3. Launch Ltxvideo

From the repository root:

~~~powershell
python run.py
~~~

The application should open at:

~~~text
http://127.0.0.1:7860
~~~

`run.py` checks the required runtime packages and prepares the local model cache when needed. The first model setup requires internet access; later launches reuse the local cache.

### 4. Start with the safest generation preset

For the first GPU smoke test, use:

~~~text
Resolution: 384×224
Frames:     49
Duration:   ~1.6 seconds at 30 FPS
~~~

If that succeeds, move to the recommended everyday starting preset:

~~~text
Resolution: 384×224
Frames:     121
Duration:   ~4 seconds at 30 FPS
~~~

### 5. If CUDA runs out of memory

First close GPU-heavy applications and reduce the generation size:

- Use `384×224`
- Try 49, 97, or 121 frames
- Restart the application after an out-of-memory failure

For additional memory and Windows troubleshooting, see the [Troubleshooting](#troubleshooting) section below.

### 6. Model cache

Model files are downloaded during the first setup and reused locally on later launches. Keep the local model cache available if you want to run without downloading the models again.

> **Note:** This checklist documents the intended first-run workflow for the project's RTX 4050 target. It does not claim that a specific generation was validated on a particular RTX 4050 machine.

## Safest first generation

For a first GPU smoke test on the target hardware, start small:

```text
Resolution: 384×224
Frames:     49
Duration:   ~1.6 s at 30 FPS
```

After that succeeds, the recommended everyday scene preset is **384×224 / 121 frames (~4 seconds)**.

## Creative Studios

| Studio | Best for | Direction added by the UI |
|---|---|---|
| ✨ **General** | Free-form T2V / I2V | Raw prompt, optional image reference, shot controls |
| 📚 **Comics** | Motion comics, manga, graphic-novel looks | Art-style and identity continuity, comic-specific negatives |
| 🌍 **Real World** | Live action, products, travel, documentary | Physical realism, materials, anatomy and natural camera language |
| ⚡ **Action** | Sports, chases, parkour, martial arts | Intensity, timing, choreography, momentum and trajectory constraints |
| 🧸 **Cartoon Story** | Connected animated scenes | Character bible, style lock, last-frame continuation and scene stitching |

Every directed studio supports subject/action/environment guidance, style, framing, camera movement, lighting, continuity notes, negative prompts, seed/guidance/step controls and a **Director Prompt Preview**.

## Prompt-director approach

LTX generally responds better to a chronological shot description than a loose pile of tags. Ltxvideo compiles directed prompts in roughly this order:

```text
Reference anchor (when supplied)
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

For short clips, keep the scene readable: **one main action beat + one camera move** is usually safer than trying to fit an entire sequence into one generation.

## RTX 4050 memory strategy

<p align="center">
  <img src="docs/assets/hardware.svg" alt="RTX 4050 memory strategy" width="100%" />
</p>

The low-memory path uses the 2B LTX-Video Diffusers architecture with quantized transformer/text-encoder loading, GPU/CPU balancing, VAE tiling/slicing and short native clips so the application does not need to keep an entire long decoded movie in memory.

```text
Studio / prompt / reference image
              │
              ▼
      Video Skill Engine
              │
              ▼
  8-bit text encoder ─────┐
                          │ GPU/CPU placement
  8-bit LTX transformer ──┤ RTX 4050 target
                          │
  tiled / sliced VAE ─────┘
              │
              ▼
         native MP4
              │
       ┌──────┴──────┐
       │             │
   delivery       cartoon story
    export        continuation
```

## Duration and resolution guidance

The UI exposes valid `8k+1` frame counts from **49 to 241 frames**. At 30 FPS that is about **1.6 to 8.0 seconds per native shot**.

| Goal | Resolution | Frames | Approx. duration | Notes |
|---|---:|---:|---:|---|
| First GPU test | 384×224 | 49 | 1.6 s | Safest smoke test |
| Recommended scene | 384×224 | 121 | 4.0 s | Default starting point |
| Better framing | 512×288 | 121 | 4.0 s | More memory pressure |
| Longer shot | 384×224 | 193 | 6.4 s | Heavy |
| Practical UI maximum | 384×224 | 241 | 8.0 s | Highest exposed native option |

Longer content should be composed from multiple shots. Cartoon Story Studio automates that pattern for animation.

## Cartoon continuity workflow

<p align="center">
  <img src="docs/assets/workflow.svg" alt="Cartoon story continuity workflow" width="100%" />
</p>

1. Write a full story or one scene beat per line.
2. Define a **Character Bible** with face, colors, clothing, props and identifying details.
3. Optionally upload a character/style reference for Scene 1.
4. Generate Scene 1.
5. Extract its last frame and use that as the visual reference for Scene 2.
6. Re-inject the same character bible and visual style into every scene prompt.
7. Repeat and join the short clips with FFmpeg.

This does not guarantee perfect identity preservation, but it gives the model a consistent visual anchor instead of restarting each scene from unrelated text.

## Export sizes

Native generation stays intentionally conservative. The delivery layer can output:

- Native MP4
- 1280×720
- 1920×1080
- 720×1280
- 1080×1920
- 1080×1080

The 720p/1080p choices are **FFmpeg delivery resizes**, not native 1080p diffusion and not AI super-resolution.

## Repository layout

```text
Ltxvideo/
├── app.py
├── run.py
├── config.py
├── download_models.py
├── requirements.txt
├── engine/
│   ├── generator.py
│   ├── prompt_builders.py
│   ├── skill_engine.py
│   ├── storyboard.py
│   ├── memory_manager.py
│   ├── video_processor.py
│   └── video_qc.py
├── static/
├── docs/
├── tests/
├── CONTRIBUTING.md
├── LICENSE
└── NOTICE
```

## Testing

Run the lightweight tests without loading the video model:

```bash
python -m unittest discover -s tests -v
```

Syntax check:

```bash
python -m py_compile app.py config.py run.py download_models.py engine/*.py tests/*.py
```

A full end-to-end generation test still requires a compatible NVIDIA CUDA GPU and downloaded model weights.

## Troubleshooting

**CUDA out of memory**  
Close GPU-heavy applications, fall back to `384×224` and 49/97/121 frames, then restart the application to clear memory.

**Windows becomes slow while loading**  
Close browsers/games and make sure the Windows page file is enabled. Sixteen GB of system RAM is a tight environment for modern video diffusion.

**Action becomes chaotic**  
Reduce the shot to one primary action beat and one camera move. Generate the next beat as a separate shot.

**Reference-image identity drifts**  
Describe what should move or change rather than re-describing the entire still image, keep continuity notes specific, and use a stable seed when comparing iterations.

## Contributing

Contributions are welcome. Start with **[CONTRIBUTING.md](CONTRIBUTING.md)** or pick one of the newcomer tasks:

- **[#4 — Windows RTX 4050 setup checklist](https://github.com/logeshv586-code/Ltxvideo/issues/4)**
- **[#5 — Copy-ready example prompts](https://github.com/logeshv586-code/Ltxvideo/issues/5)**
- **[#6 — Lightweight startup diagnostics](https://github.com/logeshv586-code/Ltxvideo/issues/6)**

All three are labeled `good first issue` and `help wanted`.

## Model and upstream projects

- [Lightricks/LTX-Video](https://github.com/Lightricks/LTX-Video)
- [Lightricks/LTX-2](https://github.com/Lightricks/LTX-2)
- [LTX-Video on Hugging Face](https://huggingface.co/Lightricks/LTX-Video)
- [Hugging Face Diffusers LTX documentation](https://huggingface.co/docs/diffusers/api/pipelines/ltx_video)

## License

The application code in this repository is licensed under the **Apache License 2.0**. See [LICENSE](LICENSE) and [NOTICE](NOTICE).

LTX model weights, upstream model/runtime components and other third-party dependencies remain subject to their own licenses and terms. Review the current upstream model license before redistribution or commercial use.

---

If this project helps you run local AI video on constrained hardware, consider **starring the repository** and sharing a reproducible generation result. Real hardware reports are especially useful to the project.
