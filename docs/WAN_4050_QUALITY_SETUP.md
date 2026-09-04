# Wan2.1 GGUF RTX 4050 Quality Setup

This is the recommended local path for an **RTX 4050 Laptop GPU with 6 GB VRAM and 16 GB system RAM**.

## Recommended backend

Use **Wan2.1-T2V-1.3B with a GGUF Q5_0 transformer**.

The application now selects the backend automatically:

```text
VRAM below 10 GB  -> GGUF transformer
VRAM 10 GB+       -> full Diffusers transformer
```

You can override this with:

```powershell
$env:WAN_BACKEND="gguf"
$env:WAN_BACKEND="full"
```

For the RTX 4050, keep `gguf`.

## Why Q5_0

Q5_0 is used as the default quality/memory compromise. It is more conservative than full FP16 weights while retaining more transformer precision than Q4_0. The remaining Wan components still come from the official Diffusers repository.

GGUF only replaces the diffusion transformer. Tokenizer, text encoder, scheduler and VAE are downloaded from:

```text
Wan-AI/Wan2.1-T2V-1.3B-Diffusers
```

The default quantized transformer comes from:

```text
samuelchristlie/Wan2.1-T2V-1.3B-GGUF
Wan2.1-T2V-1.3B-Q5_0.gguf
```

Both locations can be overridden with environment variables if needed.

## One-time Windows setup

From the repository folder:

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python setup_wan.py --download-gguf
```

Or use:

```text
setup_wan_4050.bat
```

The GGUF setup downloads the official pipeline components but skips the full transformer safetensors, then downloads only the Q5_0 GGUF transformer.

Check the setup:

```powershell
python setup_wan.py --status
```

## Recommended 4050 runtime

```powershell
$env:WAN_BACKEND="gguf"
$env:WAN_OFFLOAD_MODE="sequential"
$env:WAN_DELIVERY_FPS="32"
$env:WAN_SMOOTHING="auto"
$env:WAN_UPSCALER="auto"
$env:REALESRGAN_TILE="256"
python run.py --video-studio-ui
```

Or simply run:

```text
run_wan_4050.bat
```

## Generation settings

Keep these defaults for the 6 GB laptop:

```text
Model       Wan2.1-T2V-1.3B
Transformer GGUF Q5_0
Native size 832 x 480
Frames      81
Native FPS  16
Steps       40
Guidance    6.0
Offload     Sequential CPU offload
VAE         Tiling + slicing, fp32 decode when possible
Workers     1
```

Do not try to make the diffusion stage render native 1080p on a 6 GB GPU. Quality is improved after generation.

## Delivery pipeline

```text
Wan2.1 GGUF Q5_0
832x480 @ 16 FPS
        |
        v
RIFE smoothing when installed
16 FPS -> 32 FPS
        |
        v
Real-ESRGAN anime/video upscaling when installed
2x, tile 256
        |
        v
FFmpeg final delivery
1920x1080 @ 32 FPS
```

When RIFE is not installed, the app falls back to FFmpeg motion interpolation. When Real-ESRGAN is not installed, it falls back to Lanczos scaling with light sharpening.

## 16 GB RAM recommendation

16 GB physical RAM is usable but tight because the Wan text encoder still consumes system memory while the transformer is quantized. Keep Windows virtual memory/page file enabled and preferably system-managed on an SSD.

While generating:

- use one GPU worker only;
- close games and GPU-heavy applications;
- close other local AI models;
- avoid running a second video generation at the same time;
- keep enough free SSD space for model files, page file and temporary frames.

## Optional RIFE and Real-ESRGAN

The app automatically detects:

```text
rife-ncnn-vulkan
realesrgan-ncnn-vulkan
```

You can set explicit Windows paths:

```powershell
$env:RIFE_EXE="C:\AI\rife\rife-ncnn-vulkan.exe"
$env:REALESRGAN_EXE="C:\AI\realesrgan\realesrgan-ncnn-vulkan.exe"
```

Keep each executable together with the model folders supplied in its release archive.

## Quality controls

Default smoothing:

```text
WAN_SMOOTHING=auto
```

Options:

```text
auto
rife
ffmpeg
off
```

Default upscaler:

```text
WAN_UPSCALER=auto
```

Real-ESRGAN defaults:

```text
model: realesr-animevideov3
scale: 2x
tile: 256
```

If Real-ESRGAN runs out of memory, reduce the tile:

```powershell
$env:REALESRGAN_TILE="192"
```

or:

```powershell
$env:REALESRGAN_TILE="128"
```

## Kids cartoon workflow

For the best consistency, generate short scenes rather than one long diffusion pass:

```text
story
  -> scene 1
  -> scene 2
  -> scene 3
  -> ...
  -> smooth/upscale each clip
  -> join clips
  -> final narration/audio
```

The story generator reuses one base seed and repeats the same character identity, face, proportions, costume, colors and art-style lock in every scene prompt.

Use one important visible action and one controlled camera movement per scene. GGUF reduces memory use, but it does not fix an overloaded or contradictory prompt.

## Useful commands

Status:

```powershell
python setup_wan.py --status
```

Download low-memory GGUF setup:

```powershell
python setup_wan.py --download-gguf
```

Download full Diffusers model for higher-VRAM machines:

```powershell
python setup_wan.py --download
```

Run the 4050 profile:

```powershell
run_wan_4050.bat
```

## Important expectation

GGUF primarily reduces model-weight memory. It does not make the RTX 4050 equivalent to a high-end 16-24 GB GPU. Generation can still be slow because sequential CPU offload moves model blocks between RAM and VRAM. The goal of this profile is **reliable local generation with the best practical cartoon quality on 6 GB VRAM**, followed by dedicated smoothing and upscaling.
