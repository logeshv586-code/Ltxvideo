<p align="center">
  <img src="docs/assets/hero.svg" alt="LTX Video Director Studio" width="100%" />
</p>

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white" />
  <img alt="GPU" src="https://img.shields.io/badge/GPU-Adaptive%20NVIDIA-76B900?logo=nvidia&logoColor=white" />
  <img alt="Multi GPU" src="https://img.shields.io/badge/Multi--GPU-Worker%20Pool-8B5CF6" />
  <img alt="Server" src="https://img.shields.io/badge/Server-2%C3%97T4%20Ready-0EA5E9" />
  <img alt="Local" src="https://img.shields.io/badge/Generation-Local%20%2F%20Offline-06B6D4" />
  <img alt="License" src="https://img.shields.io/github/license/logeshv586-code/Ltxvideo" />
  <img alt="Stars" src="https://img.shields.io/github/stars/logeshv586-code/Ltxvideo?style=flat" />
</p>

# LTX Video Director Studio

**Adaptive local AI video generation for NVIDIA GPUs — from low-VRAM laptops to multi-GPU servers.**

Ltxvideo wraps Lightricks LTX-Video in a Gradio-based video studio that automatically detects the machine it is running on and adjusts generation behavior from the available **GPU model, GPU count, VRAM per GPU and system RAM**.

The project supports text-to-video, image-to-video, continuous multi-scene generation, cartoons, action, realistic scenes and delivery exports while keeping generation memory-aware.

> **No paid cloud video-generation API is required after the first model download.** Model files are cached locally and reused on later runs.

---

## Highlights

- **Automatic hardware detection** — GPU model, CUDA device count, per-GPU VRAM and system RAM.
- **Adaptive native quality** — higher-VRAM GPUs automatically render at larger native sizes when safe.
- **Multi-GPU worker pool** — multiple cards are used as independent generation workers.
- **Dual NVIDIA T4 server support** — tested architecture for `2 × T4 16 GB` style deployments.
- **Remote Gradio UI** — run generation on a Linux GPU server and control it from your own computer.
- **Text → Video** and **Image → Video** generation.
- **Continuous Video mode** — longer videos are generated part-by-part and joined automatically.
- **Character / visual continuity support** using previous-scene conditioning.
- **Automatic visual QC and retry path** for failed or unstable generated clips.
- **720p / 1080p delivery exports** for landscape, portrait and square formats.
- **Low-memory fallback profiles** for RTX 3050 / RTX 4050 class hardware.

---

## Adaptive hardware architecture

Ltxvideo no longer assumes every machine is an RTX 4050.

At startup it detects:

```text
GPU count
   ↓
GPU model for every device
   ↓
VRAM available on each GPU
   ↓
System RAM
   ↓
Hardware profile selection
   ↓
GPU / CPU memory budget
   ↓
Native quality scaling
   ↓
Generation worker count
```

Each physical GPU remains an independent device.

For example:

```text
2 × NVIDIA T4 16 GB

        ┌──────────────────┐
        │ Gradio job queue │
        └────────┬─────────┘
                 │
       ┌─────────┴─────────┐
       │                   │
       ▼                   ▼
┌─────────────┐      ┌─────────────┐
│ GPU Worker 0│      │ GPU Worker 1│
│ T4 · 16 GB  │      │ T4 · 16 GB  │
└──────┬──────┘      └──────┬──────┘
       │                    │
       ▼                    ▼
 LTX pipeline          LTX pipeline
       │                    │
       ▼                    ▼
 Video request A       Video request B
```

The two cards are **not treated as one fake 32 GB GPU**. Each render stays inside one GPU's VRAM limit, while separate requests can run concurrently.

---

## Hardware profiles

The application chooses a conservative profile automatically.

| Hardware | Typical behavior |
|---|---|
| 4 GB NVIDIA GPU | Low-VRAM safe profile, short clips and conservative resolution |
| 6 GB RTX 3050 / RTX 4050 | Balanced low-memory generation with CPU offload |
| 8–12 GB GPU | Increased native resolution when safe |
| 16 GB GPU | Server-quality profile with larger native generation |
| NVIDIA T4 16 GB | Dedicated T4 profile with roughly 14 GiB runtime GPU budget |
| 20 GB+ GPU | Higher-quality adaptive profile with larger spatial limits |
| Multiple GPUs | One persistent generation worker per visible GPU, up to configured limit |

System RAM also changes the amount of memory available for CPU offload. Higher-RAM servers therefore receive a larger CPU memory budget automatically.

---

# Quick start

## Windows / local workstation

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python run.py
```

Open:

```text
http://127.0.0.1:7860
```

---

## Linux local workstation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python run.py
```

---

# Dual NVIDIA T4 server setup

For a server with **2 × NVIDIA T4 16 GB**, use the included server launcher.

```bash
git clone https://github.com/logeshv586-code/Ltxvideo.git
cd Ltxvideo
chmod +x server_start.sh
./server_start.sh
```

The server UI listens on:

```text
http://0.0.0.0:7860
```

From your own computer, open:

```text
http://SERVER_IP:7860
```

Before generation, verify the hardware:

```bash
nvidia-smi
python run.py --check
```

A two-T4 system should report approximately:

```text
Hardware profile: NVIDIA T4 16 GB server profile
GPU 0: Tesla T4 · ~15 GB VRAM
GPU 1: Tesla T4 · ~15 GB VRAM
System RAM: ... GB
Adaptive GPU workers: 2
```

Full deployment notes are available in:

**[docs/T4_SERVER_SETUP.md](docs/T4_SERVER_SETUP.md)**

---

## Server launch options

### Normal server mode

```bash
python run.py --server
```

### Custom port

```bash
python run.py --server --port 8080
```

### Restrict generation to one GPU worker

```bash
LTX_MAX_GPU_WORKERS=1 python run.py --server
```

### Use two GPU workers

```bash
LTX_MAX_GPU_WORKERS=2 python run.py --server
```

### Override host

```bash
LTX_SERVER_NAME=0.0.0.0 python run.py --server
```

### Override memory budgets

```bash
LTX_GPU_MEMORY_BUDGET=13GiB \
LTX_CPU_MEMORY_BUDGET=28GiB \
python run.py --server
```

Normally these overrides are unnecessary because the application detects the hardware automatically.

---

# Video generation modes

## Easy Video Creator

The default interface is designed for normal video creation without requiring the user to understand diffusion settings.

```bash
python run.py
```

Available workflows include:

- Single Clip
- Continuous Video
- Text-to-video
- Image-to-video
- Automatic style detection
- 3D Animation
- Clay Animation
- Anime
- Cinematic
- Product Video

---

## Advanced / legacy studio

```bash
python run.py --legacy-ui
```

The advanced UI exposes dedicated creative studios:

| Studio | Best for |
|---|---|
| ✨ General | Free-form T2V / I2V |
| 📚 Comics | Motion comics, manga and graphic-novel visuals |
| 🌍 Real World | Live action, products, travel and documentary scenes |
| ⚡ Action | Sports, chases, parkour and martial arts |
| 🧸 Cartoon Story | Connected animated scenes and character continuity |

---

# Adaptive video quality

The UI still presents simple quality choices such as **Balanced** and **High**.

The backend then adapts those requested dimensions to the available hardware.

Example on a higher-VRAM server GPU:

```text
UI planned size
576 × 320
      │
      ▼
Hardware profile detected
NVIDIA T4 · 16 GB
      │
      ▼
Adaptive spatial scaling
      │
      ▼
approximately 768 × 416 native render
      │
      ▼
Final delivery export
1280 × 720
```

The exact adaptive size can vary because the generator also enforces a maximum native pixel budget to protect VAE and attention memory peaks.

Longer clips receive more conservative scaling than short clips because temporal generation consumes more memory.

---

## Recommended T4 quality settings

For the **best quality / stability balance on a T4**, start with:

```text
Quality: High
Backend clip length: 4 seconds
Aspect: 16:9
Mode: Single Clip or Continuous Video
```

For longer videos, prefer many short continuation clips instead of pushing one very long diffusion render.

This gives the model more spatial detail while keeping motion and VRAM usage manageable.

---

# Continuous video generation

Long-form output is generated sequentially rather than attempting to create minutes of video in one diffusion pass.

```text
Full story
   │
   ▼
Story planner
   │
   ├── Scene 1
   ├── Scene 2
   ├── Scene 3
   └── ...

Scene 1 → generated clip
              │
              ▼
        tail frames extracted
              │
              ▼
Scene 2 → conditioned generation
              │
              ▼
        tail frames extracted
              │
              ▼
Scene 3 → conditioned generation
              │
              ▼
             ...
              │
              ▼
       FFmpeg concatenation
              │
              ▼
         final video
```

This workflow helps preserve:

- character identity
- colors and clothing
- camera direction
- motion progression
- environment continuity

Perfect identity consistency is not guaranteed, but continuation conditioning is considerably more stable than restarting every scene from text only.

---

# Prompt-director approach

LTX generally performs better with a chronological visual description rather than a large collection of unrelated prompt tags.

Ltxvideo builds prompts approximately in this order:

```text
Reference anchor
      ↓
Subject
      ↓
Visible action
      ↓
Environment
      ↓
Style
      ↓
Shot framing
      ↓
Camera motion
      ↓
Lighting
      ↓
Continuity rules
      ↓
Stability / anatomy / physics constraints
```

For short clips, the safest approach remains:

> **One important action + one understandable camera move per generated shot.**

---

# Quality control

Generated clips pass through lightweight video QC.

The system can detect problems such as:

- unreadable or missing output
- incorrect dimensions
- incorrect duration
- frozen / near-static results
- severe visual instability
- failed video encoding

When a visual-quality failure is detected, the generation workflow can retry using another seed or a stable continuation frame.

For CUDA out-of-memory errors, the worker:

1. unloads its current pipeline,
2. clears the cache for that GPU,
3. reloads the model,
4. retries once.

---

# Export sizes

The delivery layer supports:

- Native MP4
- 1280×720 landscape
- 1920×1080 landscape
- 720×1280 portrait
- 1080×1920 portrait
- 1080×1080 square

Important: the delivery resolution and the diffusion generation resolution are different concepts.

A 1080p export does **not** mean the diffusion model generated every frame natively at 1920×1080. The current delivery stage uses high-quality FFmpeg scaling and light sharpening.

The adaptive server path improves quality primarily by increasing the **native generation size before export** when VRAM allows it.

---

# Repository layout

```text
Ltxvideo/
├── app.py                     # advanced studio UI
├── easy_app.py                # simple customer-facing UI
├── run.py                     # adaptive launcher
├── server_start.sh            # Linux server bootstrap
├── config.py
├── diagnostics.py
├── download_models.py
├── requirements.txt
│
├── engine/
│   ├── generator.py           # core LTX generation
│   ├── optimized_generator.py # adaptive multi-GPU worker pool
│   ├── hardware_profiles.py   # GPU / VRAM / RAM profiles
│   ├── longform.py            # continuous-video planner/generator
│   ├── prompt_builders.py
│   ├── skill_engine.py
│   ├── storyboard.py
│   ├── memory_manager.py
│   ├── video_processor.py
│   └── video_qc.py
│
├── docs/
│   └── T4_SERVER_SETUP.md
│
├── tests/
├── outputs/
├── models/
├── CONTRIBUTING.md
├── LICENSE
└── NOTICE
```

---

# Diagnostics

Run:

```bash
python run.py --check
```

The diagnostic command does not load the full generation model. It checks the runtime environment and reports information such as:

- Python version
- PyTorch installation
- CUDA availability
- GPU model
- VRAM
- selected hardware profile
- safe generation preset
- FFmpeg
- model cache status

---

# Testing

Run the lightweight unit tests without loading the video model:

```bash
python -m unittest discover -s tests -v
```

Syntax check:

```bash
python -m py_compile app.py easy_app.py config.py run.py diagnostics.py download_models.py engine/*.py tests/*.py
```

A true end-to-end generation test requires a compatible NVIDIA CUDA GPU and the downloaded model weights.

---

# Troubleshooting

### CUDA out of memory

Try, in order:

```text
1. Use a 4-second backend clip.
2. Change High → Balanced.
3. Reduce LTX_MAX_GPU_WORKERS to 1 if system RAM is limited.
4. Close other GPU workloads.
5. Restart the generation worker/app.
```

### Two GPUs detected but only one is being used

Check:

```bash
nvidia-smi
```

Then ensure:

```bash
LTX_MAX_GPU_WORKERS=2 python run.py --server
```

Two workers allow two separate generation jobs to execute concurrently. A single video render normally remains on one GPU.

### Cannot open the server UI from another computer

Run:

```bash
python run.py --server
```

Then verify that TCP port `7860` is allowed by the firewall/security group.

Open:

```text
http://SERVER_IP:7860
```

For internet-facing deployments, place the application behind an authenticated reverse proxy, VPN or other access-control layer rather than exposing Gradio openly.

### Action becomes chaotic

Reduce the scene to one primary visible action and one camera movement. Put the next action in the next generated clip.

### Reference-image identity drifts

Describe what should **move or change** rather than repeatedly re-describing the entire reference image. Use explicit character-lock details and stable seeds while comparing generations.

---

# Model and upstream projects

- [Lightricks/LTX-Video](https://github.com/Lightricks/LTX-Video)
- [Lightricks/LTX-2](https://github.com/Lightricks/LTX-2)
- [LTX-Video on Hugging Face](https://huggingface.co/Lightricks/LTX-Video)
- [Hugging Face Diffusers LTX documentation](https://huggingface.co/docs/diffusers/api/pipelines/ltx_video)

---

# License

The application code in this repository is licensed under the **Apache License 2.0**. See [LICENSE](LICENSE) and [NOTICE](NOTICE).

LTX model weights, upstream model/runtime components and other third-party dependencies remain subject to their own licenses and terms. Review the current upstream model license before redistribution or commercial use.

---

## Current deployment target

The current server-focused configuration is especially suited for:

```text
2 × NVIDIA T4 16 GB
32–64+ GB system RAM
Linux
Python 3.10+
CUDA-enabled PyTorch
Remote browser UI
```

At the same time, the same codebase retains lower-memory profiles so development and testing can still run on smaller NVIDIA GPUs.

If this project helps you run local AI video generation, consider **starring the repository** and sharing reproducible hardware results with the GPU model, VRAM, prompt, seed, native resolution, frame count and generation time.
