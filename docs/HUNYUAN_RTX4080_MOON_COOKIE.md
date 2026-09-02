# Moon Cookie production mode — HunyuanVideo-1.5 on RTX 4080

This repository now includes a second generation backend for the Moon Cookie cartoon workflow. The existing LTX backend remains available for low-VRAM machines; the Hunyuan backend targets an RTX 4080 16 GB GPU.

## Recommended model

- HunyuanVideo-1.5 480p Image-to-Video Step-Distilled
- 12 inference steps for final shots
- 8 steps for faster review shots
- 4 steps for rough storyboard previews
- CPU + group offloading enabled
- Cache disabled when step distillation is enabled
- Official super-resolution enabled for final/review presets

The official Hunyuan repository recommends 8 or 12 steps for the 480p I2V step-distilled model. The integration therefore does not expose incompatible 50-step CFG-distilled settings in this Moon Cookie UI.

## Install

```bash
python setup_hunyuan.py --install-code
python setup_hunyuan.py --show-downloads
```

Follow the printed checkpoint commands. Hunyuan's vision encoder uses the FLUX.1-Redux-dev SigLIP model, which requires your own approved Hugging Face access/token.

Then launch:

```bash
python run.py --hunyuan-ui
```

## Episode plan

For a 2.5-minute Tamil Moon Cookie episode:

| Item | Count |
|---|---:|
| Final duration | ~150 seconds |
| Final 5-second shots | 30 |
| Attempts per shot | 2 |
| Raw generations | ~60 |

Production sequence:

1. Start Scene 1 from the best clean Moon Cookie reference image.
2. Describe one clear action only for the next ~5-second shot.
3. Generate two attempts and keep the stronger one.
4. Extract the accepted final frame and use it as the next scene reference.
5. Repeat until all 30 shots are accepted.
6. Add Tamil dialogue/character voices after visual generation.
7. Add background music and sound effects.
8. Use lip-sync only for close dialogue shots where it adds value.
9. Assemble and grade the final episode in the editor.

## Why visual-first

Generating dialogue inside every video shot increases the number of constraints the video model must satisfy simultaneously. The Moon Cookie mode therefore focuses Hunyuan on identity, motion, composition and continuity first; Tamil voices and precise lip-sync remain downstream editing steps.

## Memory notes

The launcher sets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128` and keeps official offloading enabled. `Overlap group offloading` is disabled by default because it consumes more system RAM; enable it from Advanced settings only when the machine has sufficient RAM and you want faster inference.

## Files added

- `engine/hunyuan_generator.py` — official CLI adapter and RTX 4080-safe presets
- `hunyuan_app.py` — dedicated Moon Cookie generation UI
- `setup_hunyuan.py` — official source/runtime setup helper
- `run.py --hunyuan-ui` — one-command launcher
