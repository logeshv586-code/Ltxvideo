# Wan2.1 RTX 4050 Quality Setup

This is the recommended local quality path for an RTX 4050 Laptop GPU with 6 GB VRAM.

## Recommended engine

Use **Wan2.1-T2V-1.3B** for the final kids-cartoon render on the 4050.

The generator intentionally stays at the stable Wan native shape:

- 832 × 480
- 81 frames
- 16 FPS
- 40 inference steps by default
- sequential CPU offload on GPUs below 12 GB
- VAE tiling and slicing enabled

The delivery pipeline then improves the generated frames instead of asking the diffusion model to render native 1080p, which is much less practical on 6 GB VRAM.

## Quality pipeline

```text
Wan2.1-T2V-1.3B
832x480 @ 16 FPS
        ↓
RIFE interpolation
16 FPS → 32 FPS
        ↓
Real-ESRGAN anime/video model
2x cartoon upscaling
        ↓
FFmpeg final delivery
1920x1080 @ 32 FPS
```

For story assembly, the same base seed is reused across scenes and the exact character/costume continuity lock is repeated in every scene prompt. This is intended to reduce identity and wardrobe drift.

## Optional portable enhancement tools

The application automatically detects these portable executables:

- `rife-ncnn-vulkan`
- `realesrgan-ncnn-vulkan`

They are intentionally external to the Python/PyTorch environment so Wan can keep most of the 6 GB CUDA budget for generation.

### Windows

Download the official portable releases, keep each executable together with the model folders included in its release archive, then either add the folders to `PATH` or set explicit paths.

PowerShell example:

```powershell
$env:RIFE_EXE="C:\AI\rife\rife-ncnn-vulkan.exe"
$env:REALESRGAN_EXE="C:\AI\realesrgan\realesrgan-ncnn-vulkan.exe"
python setup_wan.py --status
python run.py --video-studio-ui
```

### Linux

```bash
export RIFE_EXE=/opt/rife/rife-ncnn-vulkan
export REALESRGAN_EXE=/opt/realesrgan/realesrgan-ncnn-vulkan
python setup_wan.py --status
python run.py --video-studio-ui --server
```

## Defaults for cartoons

The integrated Real-ESRGAN path uses:

```text
model: realesr-animevideov3
scale: 2x
tile: 256
```

The 256 tile default is deliberately conservative for laptop GPUs. Override it only after a stable test:

```powershell
$env:REALESRGAN_TILE="192"   # less memory / slower
$env:REALESRGAN_TILE="320"   # potentially faster / more memory
```

## Smoothing controls

Default:

```text
WAN_SMOOTHING=auto
```

Behavior:

1. Use RIFE when the portable executable is installed.
2. Otherwise use FFmpeg motion-compensated interpolation.
3. If interpolation fails, keep native 16 FPS rather than crashing the generated video.

Overrides:

```powershell
$env:WAN_SMOOTHING="rife"
$env:WAN_SMOOTHING="ffmpeg"
$env:WAN_SMOOTHING="off"
```

The default delivery target is 32 FPS. It can be changed with:

```powershell
$env:WAN_DELIVERY_FPS="32"
```

The current low-memory RIFE integration is designed for 2× interpolation, so 32 FPS is recommended for Wan's native 16 FPS output.

## Upscaling controls

Default:

```text
WAN_UPSCALER=auto
```

Behavior:

1. Use `realesr-animevideov3` when Real-ESRGAN NCNN/Vulkan is installed.
2. Otherwise use the existing high-quality Lanczos + light sharpening delivery path.

Overrides:

```powershell
$env:WAN_UPSCALER="realesrgan"
$env:WAN_UPSCALER="off"
$env:REALESRGAN_MODEL="realesr-animevideov3"
$env:REALESRGAN_SCALE="2"
```

## Wan model setup

```powershell
python setup_wan.py --download
python setup_wan.py --status
python run.py --video-studio-ui
```

For a 6 GB RTX 4050, keep this unless a measured test proves otherwise:

```powershell
$env:WAN_OFFLOAD_MODE="sequential"
```

## What this improves

RIFE improves perceived motion smoothness. Real-ESRGAN improves edges and stylized texture at delivery resolution. Neither tool can completely repair a bad diffusion generation, so the Wan prompt also locks character identity, proportions, costume, colors and simple controlled movement.

For kids' story videos, use short scenes with one main visible action per clip. Generate several stable clips and assemble them rather than trying to create a long complex action in one diffusion pass.
