# 🎬 LTX-2.3 Video Studio

Local text-to-video and image-to-video generation using Lightricks LTX-2.3 with synchronized audio.

This branch is tuned for a **16 GB NVIDIA GPU**. It uses the distilled LTX pipeline, FP8-cast weights, automatic CPU/disk offload, and keeps the initialized model alive between generations for faster repeated runs.

## Recommended 16 GB profile

- Resolution: **512×320**
- Duration: **97 frames / ~4 seconds**
- Pipeline: **LTX-2.3 distilled**
- Quantization: **FP8-cast**
- Offload: **automatic**
  - 40 GB+ system RAM: CPU offload/cache
  - lower RAM: disk offload
- Model stays loaded after generation to reduce repeated startup cost

## Features

- Text-to-video
- Image-to-video using first-frame conditioning
- LTX-generated synchronized audio muxed directly into MP4
- Multi-clip continuation from the last frame of the previous clip
- Gradio browser UI
- Low/fast and higher-quality resolution presets

## Windows setup

### 1. Create and activate a virtual environment

```powershell
python -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Install the official LTX runtime

```powershell
python setup_ltx.py
```

`setup_ltx.py` clones the official `Lightricks/LTX-2` source into `vendor/LTX-2`, pins it to a tested upstream commit, and installs `ltx-core` + `ltx-pipelines` into the active Python environment.

### 3. Authenticate with Hugging Face

The official Gemma encoder may require accepting its model terms first.

```powershell
hf auth login
```

### 4. Download the model assets

```powershell
python download_models.py
```

To verify an existing model folder without downloading:

```powershell
python download_models.py --check
```

Required assets are:

- `ltx-2.3-22b-distilled-1.1.safetensors`
- `ltx-2.3-spatial-upscaler-x2-1.1.safetensors`
- `google/gemma-3-12b-it-qat-q4_0-unquantized`

### 5. Launch

```powershell
python app.py
```

Open `http://localhost:7860` if the browser does not open automatically.

## Fastest practical settings on 16 GB VRAM

Start with **384×256 + 49 frames** when testing prompts. Once a prompt is good, move to the recommended **512×320 + 97 frames** preset. Higher resolutions and longer clips increase generation time substantially.

Do not enable `torch.compile` by default on a 16 GB card. It can increase peak memory and makes the first run slower. This app instead prioritizes FP8-cast, distilled sampling, automatic tiling and model reuse.

## Important behavior

The distilled LTX pipeline uses its trained fixed sampling schedule. The legacy UI still accepts the old inference-step and guidance controls for compatibility, but the local distilled engine does not override the model's fixed schedule with those values.

Single-clip output preserves LTX's synchronized audio. Multi-clip continuation currently joins generated MP4 clips using the existing continuation processor; video continuity works through last-frame conditioning.

## Troubleshooting

If `python app.py` reports that `ltx_pipelines` or `ltx_core` is unavailable, run:

```powershell
python setup_ltx.py
```

If model loading reports missing assets, run:

```powershell
python download_models.py --check
```

If Hugging Face returns `401` or `403`, accept the required model terms on Hugging Face and run `hf auth login` again.

## Upstream

- LTX source: `Lightricks/LTX-2`
- LTX-2.3 model: `Lightricks/LTX-2.3`
