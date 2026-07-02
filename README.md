# 🎬 LTX-2.3 Video Studio

A local AI video generation platform powered by **Lightricks' LTX-2.3** — a 22-billion parameter DiT-based audio-video foundation model.

## ✨ Features

- **Text-to-Video (T2V)**: Generate videos from text descriptions
- **Image-to-Video (I2V)**: Animate static images with AI-driven motion
- **Multi-Clip Continuation**: Chain 10-second clips for up to 30s videos with visual continuity
- **Premium Web UI**: Dark glassmorphism Gradio interface with prompt presets
- **Memory Optimized**: Runs on GPUs with as low as 6GB VRAM via CPU offloading

## 🚀 Quick Start

### 1. Install Dependencies
```bash
# Create virtual environment
python -m venv venv
venv\Scripts\activate   # Windows

# Install PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Install other dependencies
pip install -r requirements.txt
```

### 2. Download Models (~50-60GB)
```bash
python download_models.py
```

### 3. Launch the UI
```bash
python app.py
```

Open your browser to `http://localhost:7860`

## 📐 Technical Notes

- **Resolution**: Must be divisible by 32
- **Frame Count**: Follows `8k+1` pattern (49, 97, 121, 161, 193, 241)
- **FPS**: 24 frames per second
- **Continuation**: Last frame of each clip conditions the next clip

## 🖥️ System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU VRAM  | 6 GB (with CPU offload) | 24 GB+ |
| System RAM | 16 GB | 32 GB+ |
| Storage | 100 GB free | 150 GB+ |
| Python | 3.10+ | 3.12 |
| CUDA | 12.x | 12.4+ |

## 📁 Project Structure

```
e:\video\
├── app.py                  # Gradio web UI
├── config.py               # Configuration & presets
├── download_models.py      # Model downloader
├── requirements.txt        # Dependencies
├── engine/
│   ├── generator.py        # Core T2V/I2V engine
│   ├── continuation.py     # Multi-clip system
│   ├── memory_manager.py   # VRAM optimization
│   └── video_processor.py  # Video utilities
├── static/style.css        # UI theme
├── models/                 # Downloaded weights
└── outputs/                # Generated videos
```

## 🔗 Credits

- **Model**: [Lightricks/LTX-2.3](https://huggingface.co/Lightricks/LTX-2.3)
- **Repository**: [Lightricks/LTX-2](https://github.com/Lightricks/LTX-2)
- **Paper**: [arXiv 2601.03233](https://arxiv.org/abs/2601.03233)
