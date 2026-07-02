"""
LTX-2.3 Video Studio — Premium Gradio Web Interface
A dark-themed, glassmorphism UI for local AI video generation.
"""

import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import gradio as gr
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    DEFAULT_DURATION,
    DEFAULT_GUIDANCE_SCALE,
    DEFAULT_NUM_INFERENCE_STEPS,
    DEFAULT_RESOLUTION,
    DURATION_PRESETS,
    OUTPUTS_DIR,
    PROMPT_PRESETS,
    RESOLUTION_PRESETS,
)
from engine.continuation import ContinuationGenerator
from engine.generator import VideoGenerator
from engine.memory_manager import get_system_info, get_vram_usage_str

# ──────────────────────────────────────────────
# Global State
# ──────────────────────────────────────────────
generator = VideoGenerator()
continuation_gen = ContinuationGenerator(generator)

# Load CSS
CSS_PATH = Path(__file__).parent / "static" / "style.css"
CUSTOM_CSS = CSS_PATH.read_text(encoding="utf-8") if CSS_PATH.exists() else ""


# ──────────────────────────────────────────────
# Helper Functions
# ──────────────────────────────────────────────
def get_system_status() -> str:
    """Get formatted system status string."""
    info = get_system_info()
    gpu_name = info.gpu.name if info.gpu else "No GPU"
    vram = f"{info.gpu.vram_total_gb:.1f}GB" if info.gpu else "N/A"
    ram = f"{info.ram_available_gb:.0f}/{info.ram_total_gb:.0f}GB"
    return f"🖥️ {gpu_name} | VRAM: {vram} | RAM: {ram}"


def apply_preset(preset_name: str) -> str:
    """Apply a prompt preset."""
    return PROMPT_PRESETS.get(preset_name, "")


# ──────────────────────────────────────────────
# Generation Functions
# ──────────────────────────────────────────────
def generate_t2v(
    prompt: str,
    negative_prompt: str,
    resolution: str,
    duration: str,
    num_steps: int,
    guidance_scale: float,
    seed: int,
    num_clips: int,
    progress=gr.Progress(track_tqdm=True),
):
    """Generate text-to-video with optional continuation."""
    if not prompt.strip():
        gr.Warning("Please enter a prompt!")
        return None, "⚠️ No prompt provided"

    res = RESOLUTION_PRESETS.get(resolution, RESOLUTION_PRESETS[DEFAULT_RESOLUTION])
    n_frames = DURATION_PRESETS.get(duration, DURATION_PRESETS[DEFAULT_DURATION])

    status_log = []

    def progress_callback(message: str, prog: float) -> None:
        status_log.append(f"[{datetime.now().strftime('%H:%M:%S')}] {message}")
        if prog >= 0:
            progress(prog, desc=message)

    try:
        if num_clips <= 1:
            # Single clip generation
            output_path = generator.generate_text_to_video(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=res["width"],
                height=res["height"],
                num_frames=n_frames,
                num_inference_steps=num_steps,
                guidance_scale=guidance_scale,
                seed=seed,
                progress_callback=progress_callback,
            )
        else:
            # Multi-clip continuation
            output_path = continuation_gen.generate_continuation(
                prompt=prompt,
                num_clips=num_clips,
                negative_prompt=negative_prompt,
                width=res["width"],
                height=res["height"],
                frames_per_clip=n_frames,
                num_inference_steps=num_steps,
                guidance_scale=guidance_scale,
                seed=seed,
                progress_callback=progress_callback,
            )

        status_text = "\n".join(status_log[-10:])  # Last 10 messages
        return str(output_path), f"✅ Generation complete!\n\n{status_text}"

    except Exception as e:
        error_msg = str(e)
        status_log.append(f"❌ Error: {error_msg}")
        gr.Error(f"Generation failed: {error_msg}")
        return None, "\n".join(status_log[-10:])


def generate_i2v(
    prompt: str,
    image: Image.Image | None,
    negative_prompt: str,
    resolution: str,
    duration: str,
    num_steps: int,
    guidance_scale: float,
    seed: int,
    num_clips: int,
    progress=gr.Progress(track_tqdm=True),
):
    """Generate image-to-video with optional continuation."""
    if not prompt.strip():
        gr.Warning("Please enter a prompt describing the motion!")
        return None, "⚠️ No prompt provided"

    if image is None:
        gr.Warning("Please upload an image!")
        return None, "⚠️ No image provided"

    res = RESOLUTION_PRESETS.get(resolution, RESOLUTION_PRESETS[DEFAULT_RESOLUTION])
    n_frames = DURATION_PRESETS.get(duration, DURATION_PRESETS[DEFAULT_DURATION])

    status_log = []

    def progress_callback(message: str, prog: float) -> None:
        status_log.append(f"[{datetime.now().strftime('%H:%M:%S')}] {message}")
        if prog >= 0:
            progress(prog, desc=message)

    try:
        if num_clips <= 1:
            output_path = generator.generate_image_to_video(
                prompt=prompt,
                image=image,
                negative_prompt=negative_prompt,
                width=res["width"],
                height=res["height"],
                num_frames=n_frames,
                num_inference_steps=num_steps,
                guidance_scale=guidance_scale,
                seed=seed,
                progress_callback=progress_callback,
            )
        else:
            output_path = continuation_gen.generate_continuation(
                prompt=prompt,
                num_clips=num_clips,
                negative_prompt=negative_prompt,
                width=res["width"],
                height=res["height"],
                frames_per_clip=n_frames,
                num_inference_steps=num_steps,
                guidance_scale=guidance_scale,
                seed=seed,
                first_image=image,
                progress_callback=progress_callback,
            )

        status_text = "\n".join(status_log[-10:])
        return str(output_path), f"✅ Generation complete!\n\n{status_text}"

    except Exception as e:
        error_msg = str(e)
        status_log.append(f"❌ Error: {error_msg}")
        gr.Error(f"Generation failed: {error_msg}")
        return None, "\n".join(status_log[-10:])


# ──────────────────────────────────────────────
# Build the Gradio Interface
# ──────────────────────────────────────────────
def create_app() -> gr.Blocks:
    """Build the premium Gradio interface."""
    theme = gr.themes.Base(
        primary_hue=gr.themes.colors.purple,
        secondary_hue=gr.themes.colors.blue,
        neutral_hue=gr.themes.colors.gray,
        font=gr.themes.GoogleFont("Inter"),
        font_mono=gr.themes.GoogleFont("JetBrains Mono"),
    ).set(
        body_background_fill="#0a0a0f",
        body_background_fill_dark="#0a0a0f",
        block_background_fill="rgba(20, 20, 35, 0.7)",
        block_background_fill_dark="rgba(20, 20, 35, 0.7)",
        block_border_color="rgba(255, 255, 255, 0.08)",
        block_border_color_dark="rgba(255, 255, 255, 0.08)",
        block_label_text_color="#9CA3AF",
        block_label_text_color_dark="#9CA3AF",
        block_title_text_color="#E8E8F0",
        block_title_text_color_dark="#E8E8F0",
        body_text_color="#E8E8F0",
        body_text_color_dark="#E8E8F0",
        body_text_color_subdued="#6B7280",
        body_text_color_subdued_dark="#6B7280",
        button_primary_background_fill="linear-gradient(135deg, #7C3AED, #2563EB)",
        button_primary_background_fill_dark="linear-gradient(135deg, #7C3AED, #2563EB)",
        button_primary_text_color="white",
        button_primary_text_color_dark="white",
        input_background_fill="rgba(255, 255, 255, 0.03)",
        input_background_fill_dark="rgba(255, 255, 255, 0.03)",
        input_border_color="rgba(255, 255, 255, 0.08)",
        input_border_color_dark="rgba(255, 255, 255, 0.08)",
        shadow_drop="0 4px 24px rgba(0, 0, 0, 0.3)",
        shadow_drop_lg="0 8px 48px rgba(0, 0, 0, 0.4)",
    )

    with gr.Blocks(
        title="LTX-2.3 Video Studio",
        theme=theme,
        css=CUSTOM_CSS,
    ) as app:
        # ─── Header ───
        gr.HTML("""
        <div class="header-banner" style="
            background: linear-gradient(135deg, #7C3AED, #2563EB, #06B6D4);
            padding: 24px 32px;
            border-radius: 0 0 16px 16px;
            margin-bottom: 16px;
            position: relative;
            overflow: hidden;
        ">
            <div style="position: relative; z-index: 1;">
                <div style="display: flex; align-items: center; justify-content: space-between;">
                    <div>
                        <h1 style="font-size: 28px; font-weight: 800; color: white; margin: 0; letter-spacing: -0.5px;">
                            🎬 LTX-2.3 Video Studio
                        </h1>
                        <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 4px 0 0 0;">
                            AI-Powered Video Generation • Text-to-Video • Image-to-Video • Multi-Clip Continuation
                        </p>
                    </div>
                    <div style="
                        background: rgba(0,0,0,0.2);
                        padding: 8px 16px;
                        border-radius: 8px;
                        font-family: 'JetBrains Mono', monospace;
                        font-size: 12px;
                        color: rgba(255,255,255,0.9);
                    ">
                        🖥️ Local Generation • Open Source
                    </div>
                </div>
            </div>
        </div>
        """)

        # ─── System Status Bar ───
        with gr.Row():
            system_status = gr.Textbox(
                value=get_system_status(),
                label="System Status",
                interactive=False,
                elem_classes=["gpu-monitor"],
                scale=4,
            )
            refresh_btn = gr.Button("🔄 Refresh", size="sm", scale=1)
            refresh_btn.click(fn=get_system_status, outputs=system_status)

        # ─── Main Content ───
        with gr.Tabs() as tabs:
            # ══════════════════════════════════
            # TAB 1: TEXT-TO-VIDEO
            # ══════════════════════════════════
            with gr.Tab("✨ Text to Video", id="t2v"):
                with gr.Row(equal_height=False):
                    # Left: Controls
                    with gr.Column(scale=2):
                        # Prompt Preset
                        t2v_preset = gr.Dropdown(
                            choices=[""] + list(PROMPT_PRESETS.keys()),
                            label="🎨 Prompt Presets",
                            value="",
                            interactive=True,
                        )

                        # Prompt
                        t2v_prompt = gr.Textbox(
                            label="📝 Prompt",
                            placeholder="Describe the video you want to create...\n\nTip: Be specific about motion, camera angles, lighting, and style.",
                            lines=4,
                            max_lines=8,
                        )

                        t2v_negative = gr.Textbox(
                            label="🚫 Negative Prompt (optional)",
                            placeholder="What to avoid: blurry, low quality, distorted...",
                            lines=2,
                            value="blurry, low quality, distorted, ugly, watermark",
                        )

                        # Settings
                        with gr.Accordion("⚙️ Generation Settings", open=True):
                            t2v_resolution = gr.Dropdown(
                                choices=list(RESOLUTION_PRESETS.keys()),
                                label="📐 Resolution",
                                value=DEFAULT_RESOLUTION,
                            )
                            t2v_duration = gr.Dropdown(
                                choices=list(DURATION_PRESETS.keys()),
                                label="⏱️ Duration per Clip",
                                value=DEFAULT_DURATION,
                            )
                            t2v_clips = gr.Slider(
                                minimum=1,
                                maximum=3,
                                step=1,
                                value=1,
                                label="🔗 Number of Clips (continuation)",
                                info="3 clips × 5s = 15s total video with visual continuity",
                            )

                        with gr.Accordion("🔧 Advanced Settings", open=False):
                            t2v_steps = gr.Slider(
                                minimum=4,
                                maximum=50,
                                step=1,
                                value=DEFAULT_NUM_INFERENCE_STEPS,
                                label="🔄 Inference Steps",
                                info="8 for distilled model (fast), 20-30 for dev model (quality)",
                            )
                            t2v_guidance = gr.Slider(
                                minimum=1.0,
                                maximum=10.0,
                                step=0.1,
                                value=DEFAULT_GUIDANCE_SCALE,
                                label="🎯 Guidance Scale (CFG)",
                                info="Higher = more prompt adherence. 3.0 recommended.",
                            )
                            t2v_seed = gr.Number(
                                label="🎲 Seed (-1 for random)",
                                value=-1,
                                precision=0,
                            )

                        # Generate Button
                        t2v_generate_btn = gr.Button(
                            "🚀 Generate Video",
                            variant="primary",
                            size="lg",
                            elem_classes=["generate-btn"],
                        )

                    # Right: Output
                    with gr.Column(scale=3):
                        t2v_output = gr.Video(
                            label="🎬 Generated Video",
                            autoplay=True,
                            height=420,
                        )
                        t2v_status = gr.Textbox(
                            label="📊 Generation Log",
                            lines=6,
                            interactive=False,
                            elem_classes=["status-text"],
                        )

                # Wire up preset dropdown
                t2v_preset.change(
                    fn=apply_preset,
                    inputs=t2v_preset,
                    outputs=t2v_prompt,
                )

                # Wire up generate button
                t2v_generate_btn.click(
                    fn=generate_t2v,
                    inputs=[
                        t2v_prompt,
                        t2v_negative,
                        t2v_resolution,
                        t2v_duration,
                        t2v_steps,
                        t2v_guidance,
                        t2v_seed,
                        t2v_clips,
                    ],
                    outputs=[t2v_output, t2v_status],
                )

            # ══════════════════════════════════
            # TAB 2: IMAGE-TO-VIDEO
            # ══════════════════════════════════
            with gr.Tab("🖼️ Image to Video", id="i2v"):
                with gr.Row(equal_height=False):
                    # Left: Controls
                    with gr.Column(scale=2):
                        i2v_image = gr.Image(
                            label="📷 Upload Image",
                            type="pil",
                            height=220,
                        )

                        i2v_prompt = gr.Textbox(
                            label="📝 Motion Prompt",
                            placeholder="Describe the MOTION you want, not the image content.\n\nExample: 'The camera slowly zooms in, flowers sway gently in the wind'",
                            lines=3,
                        )

                        i2v_negative = gr.Textbox(
                            label="🚫 Negative Prompt",
                            value="blurry, low quality, distorted, ugly, watermark",
                            lines=2,
                        )

                        with gr.Accordion("⚙️ Settings", open=True):
                            i2v_resolution = gr.Dropdown(
                                choices=list(RESOLUTION_PRESETS.keys()),
                                label="📐 Resolution",
                                value=DEFAULT_RESOLUTION,
                            )
                            i2v_duration = gr.Dropdown(
                                choices=list(DURATION_PRESETS.keys()),
                                label="⏱️ Duration",
                                value=DEFAULT_DURATION,
                            )
                            i2v_clips = gr.Slider(
                                minimum=1, maximum=3, step=1, value=1,
                                label="🔗 Continuation Clips",
                            )

                        with gr.Accordion("🔧 Advanced", open=False):
                            i2v_steps = gr.Slider(
                                minimum=4, maximum=50, step=1,
                                value=DEFAULT_NUM_INFERENCE_STEPS,
                                label="Steps",
                            )
                            i2v_guidance = gr.Slider(
                                minimum=1.0, maximum=10.0, step=0.1,
                                value=DEFAULT_GUIDANCE_SCALE,
                                label="Guidance Scale",
                            )
                            i2v_seed = gr.Number(label="Seed", value=-1, precision=0)

                        i2v_generate_btn = gr.Button(
                            "🚀 Generate from Image",
                            variant="primary",
                            size="lg",
                            elem_classes=["generate-btn"],
                        )

                    # Right: Output
                    with gr.Column(scale=3):
                        i2v_output = gr.Video(
                            label="🎬 Generated Video",
                            autoplay=True,
                            height=420,
                        )
                        i2v_status = gr.Textbox(
                            label="📊 Generation Log",
                            lines=6,
                            interactive=False,
                            elem_classes=["status-text"],
                        )

                i2v_generate_btn.click(
                    fn=generate_i2v,
                    inputs=[
                        i2v_prompt,
                        i2v_image,
                        i2v_negative,
                        i2v_resolution,
                        i2v_duration,
                        i2v_steps,
                        i2v_guidance,
                        i2v_seed,
                        i2v_clips,
                    ],
                    outputs=[i2v_output, i2v_status],
                )

            # ══════════════════════════════════
            # TAB 3: ABOUT / HELP
            # ══════════════════════════════════
            with gr.Tab("ℹ️ About", id="about"):
                gr.Markdown("""
                ## 🎬 LTX-2.3 Video Studio

                **Powered by Lightricks' LTX-2.3** — a 22-billion parameter DiT-based
                audio-video foundation model.

                ### 🚀 Features
                - **Text-to-Video**: Generate videos from text descriptions
                - **Image-to-Video**: Animate static images with AI-driven motion
                - **Multi-Clip Continuation**: Chain clips for up to 30s videos with visual continuity
                - **Prompt Presets**: Quick-start with curated prompt templates
                - **Memory Optimized**: Runs on GPUs with as low as 6GB VRAM

                ### 📐 Technical Details
                - **Model**: LTX-2.3 22B Distilled (8-step inference)
                - **Resolution**: Must be divisible by 32
                - **Frame Count**: Follows 8k+1 pattern (49, 97, 121, 161, 193, 241...)
                - **FPS**: 24 frames per second
                - **Continuation**: Last frame of each clip conditions the next clip

                ### ⚙️ Memory Optimization Stack
                1. Sequential CPU Offloading
                2. VAE Slicing & Tiling
                3. Attention Slicing
                4. Expandable CUDA memory segments

                ### 🔗 Links
                - [GitHub: Lightricks/LTX-2](https://github.com/Lightricks/LTX-2)
                - [HuggingFace: Lightricks/LTX-2.3](https://huggingface.co/Lightricks/LTX-2.3)
                - [Paper: arXiv 2601.03233](https://arxiv.org/abs/2601.03233)

                ### 💡 Tips
                - **For I2V**: Describe the *motion* you want, not the image content
                - **Low VRAM**: Use "Low" resolution and fewer frames
                - **Better quality**: Increase inference steps (20-30) and use "High" resolution
                - **Reproducibility**: Set a fixed seed value
                """)

    return app


# ──────────────────────────────────────────────
# Main Entry Point
# ──────────────────────────────────────────────
if __name__ == "__main__":
    print()
    print("╔══════════════════════════════════════════════════╗")
    print("║     🎬 LTX-2.3 Video Studio                     ║")
    print("║     Starting Gradio server...                    ║")
    print("╚══════════════════════════════════════════════════╝")
    print()

    # Print system info
    from engine.memory_manager import print_system_report
    print_system_report()

    app = create_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        inbrowser=True,
        show_error=True,
    )
