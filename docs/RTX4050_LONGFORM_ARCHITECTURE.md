# RTX 4050 6 GB Long-Form Video Architecture

## Decision

For a laptop-class RTX 4050 with 6 GB VRAM, the default local engine should remain in the LTX 2B family. Larger open video models can be offered later as remote/high-VRAM backends, but they should not be the default local path.

## Why these repositories were evaluated

### Lightricks/LTX-Video and Lightricks/LTX-2

LTX is the best architectural fit for this project because it is explicitly designed around efficient video generation, image-to-video continuity, keyframes, video extension and low-VRAM variants. LTX-Video 0.9.8 includes a 2B distilled model intended for light VRAM usage. LTX-2/2.5 raises quality and adds synchronized audio, but the current 22B-class model is far too large to be a practical 6 GB laptop default.

### Wan-Video/Wan2.2

Wan2.2 is a strong quality reference and should remain a future optional backend. Its official 14B models require workstation/server-class memory, while even the 5B 720p path targets much larger consumer GPUs than a 6 GB laptop. It is therefore not selected as the default local engine.

### higgsfield-ai/higgsfield

This repository is GPU orchestration / distributed ML infrastructure rather than an end-user local text-to-video inference model. Its fault-tolerance and job-orchestration ideas are useful for a future multi-GPU/cloud scheduler, but it does not replace the local LTX renderer.

### Anil-matcha/Open-Generative-AI

This repository is useful as product/UI inspiration: a creator should see a simple prompt-driven studio rather than model internals. It is not used as the core local inference engine.

## New product flow

1. User pastes one story, paragraph or script.
2. User selects visual style, publishing destination, desired duration and a simple quality/speed mode.
3. The planner estimates duration automatically from paragraph length when Auto is selected.
4. The story is split into chronological visible-action beats.
5. Each beat is expanded into an LTX-friendly shot prompt with continuity, camera and stable end-pose instructions.
6. Scenes are generated sequentially so only one short clip needs GPU memory at a time.
7. The last frame of scene N becomes the image condition for scene N+1.
8. Clips are concatenated through FFmpeg streaming.
9. A final YouTube/Reels/Square delivery MP4 is produced.

## Customer quality modes

The UI deliberately hides frames, diffusion steps and native render dimensions.

- Fast: draft/long-story mode. Lowest native resolution, fewer steps, 720p-class delivery upscale.
- Balanced: recommended RTX 4050 6 GB mode. 512x288 landscape native generation, longer coherent shots, 1080p delivery upscale.
- Quality: 576x320 landscape native generation and more inference work. Slower and more likely to stress 6 GB VRAM, but produces more native detail.

Portrait and square versions use equivalent LTX-aligned dimensions divisible by 32.

## Performance changes

The optimized generator keeps the existing proven 8-bit low-memory Diffusers loader, but removes the unconditional `torch.cuda.empty_cache()` from every successful inference call. Cache clearing is now reserved for real CUDA memory recovery. `torch.inference_mode()`, TF32 matmul hints and cuDNN benchmarking are enabled where supported.

Long-form rendering is intentionally serialized. Parallel scene generation would usually cause OOM on 6 GB VRAM.

## Model roadmap

### Default now

LTX 2B low-memory / 8-bit Diffusers path. This is the safest Windows-friendly path already integrated in the repository.

### Next performance experiment

LTXV 2B 0.9.8 distilled + Ada Q8 kernels can be benchmarked as an optional Linux/CUDA backend. Community Q8 work reports large memory and speed improvements on Ada GPUs, but it requires custom kernels/patches and should not silently replace the stable Windows path until it passes RTX 4050 testing.

### Not default on 6 GB

- LTX-2.5 22B: quality tier for much larger GPUs/cloud.
- Wan2.2 5B/14B: high-VRAM optional backend.
- LTXV 13B: high-VRAM optional backend.

## Long-video reality

A 4-5 minute AI-generated video on a 6 GB laptop is technically possible through sequential scene generation, but it is not a single inference. It can require roughly 60-95 short generations depending on quality mode. The new architecture makes this memory-safe and automatic; total wall-clock time still scales with the number of scenes.
