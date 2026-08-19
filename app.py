"""LTX Cartoon Studio — premium local UI for RTX 4050 laptops."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import gradio as gr

from config import (
    CARTOON_STYLES,
    DEFAULT_DURATION,
    DEFAULT_GUIDANCE_SCALE,
    DEFAULT_NUM_INFERENCE_STEPS,
    DEFAULT_RESOLUTION,
    DEFAULT_STORY_SCENES,
    DURATION_PRESETS,
    EXPORT_PRESETS,
    MAX_STORY_SCENES,
    NEGATIVE_PROMPT,
    OUTPUTS_DIR,
    RESOLUTION_PRESETS,
)
from engine.generator import VideoGenerator
from engine.memory_manager import get_status_markdown
from engine.storyboard import CartoonStoryGenerator, storyboard_markdown
from engine.video_processor import export_delivery

GENERATOR = VideoGenerator()
STORY_GENERATOR = CartoonStoryGenerator(GENERATOR)
CSS_PATH = Path(__file__).parent / "static" / "style.css"
CUSTOM_CSS = CSS_PATH.read_text(encoding="utf-8") if CSS_PATH.exists() else ""


def _resolve(resolution: str, duration: str) -> tuple[int, int, int]:
    res = RESOLUTION_PRESETS[resolution]
    return res["width"], res["height"], DURATION_PRESETS[duration]


def _delivery(path: Path, preset: str) -> Path:
    target = EXPORT_PRESETS.get(preset)
    if target is None:
        return path
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = OUTPUTS_DIR / f"delivery_{target[0]}x{target[1]}_{stamp}.mp4"
    return export_delivery(path, out, *target)


def _progress_bridge(progress, logs: list[str]):
    def callback(message: str, value: float) -> None:
        logs.append(message)
        if value >= 0:
            progress(min(1.0, max(0.0, value)), desc=message)
    return callback


def generate_single(prompt, image, negative, resolution, duration, steps, guidance, seed, export_preset, progress=gr.Progress()):
    if not prompt or not prompt.strip():
        raise gr.Error("Enter a prompt first.")
    width, height, frames = _resolve(resolution, duration)
    logs: list[str] = []
    callback = _progress_bridge(progress, logs)
    if image is None:
        path = GENERATOR.generate_text_to_video(prompt, negative, width, height, frames, steps, guidance, int(seed), callback)
    else:
        path = GENERATOR.generate_image_to_video(prompt, image, negative, width, height, frames, steps, guidance, int(seed), callback)
    final = _delivery(Path(path), export_preset)
    return str(final), "\n".join(logs[-12:])


def preview_story(story, scene_count, style, character_bible):
    return storyboard_markdown(story, int(scene_count), style, character_bible)


def generate_story(story, character_bible, style, scene_count, reference_image, resolution, duration, steps, guidance, seed, negative, export_preset, progress=gr.Progress()):
    if not story or not story.strip():
        raise gr.Error("Write the story or one scene beat per line.")
    width, height, frames = _resolve(resolution, duration)
    logs: list[str] = []
    callback = _progress_bridge(progress, logs)
    path = STORY_GENERATOR.generate(
        story=story,
        character_bible=character_bible,
        style_name=style,
        scene_count=int(scene_count),
        reference_image=reference_image,
        width=width,
        height=height,
        frames_per_scene=frames,
        num_inference_steps=int(steps),
        guidance_scale=float(guidance),
        negative_prompt=negative,
        seed=int(seed),
        progress_callback=callback,
    )
    final = _delivery(Path(path), export_preset)
    total = (frames / 30.0) * int(scene_count)
    logs.append(f"Approx story duration: {total:.1f}s before joins/encoding adjustments")
    return str(final), "\n".join(logs[-20:])


def create_app() -> gr.Blocks:
    theme = gr.themes.Base(
        primary_hue=gr.themes.colors.violet,
        secondary_hue=gr.themes.colors.cyan,
        neutral_hue=gr.themes.colors.slate,
    )

    with gr.Blocks(title="LTX Cartoon Studio", theme=theme, css=CUSTOM_CSS) as app:
        gr.HTML("""
        <section class="hero-shell">
          <div class="hero-kicker">LOCAL • OFFLINE AFTER FIRST DOWNLOAD • RTX 4050 PROFILE</div>
          <h1>LTX <span>Cartoon Studio</span></h1>
          <p>Create cinematic clips, animate references, and build continuous cartoon stories scene-by-scene on a laptop GPU.</p>
          <div class="hero-pills"><b>2B LTX</b><b>8-bit memory mode</b><b>Story continuity</b><b>16:9 • 9:16 • 1:1</b></div>
        </section>
        """)

        with gr.Row():
            hardware = gr.Markdown(get_status_markdown(), elem_classes=["status-card"])
            refresh = gr.Button("Refresh hardware", elem_classes=["soft-btn"])
            refresh.click(get_status_markdown, outputs=hardware)

        with gr.Tabs():
            with gr.Tab("🎬 Create Video"):
                with gr.Row(equal_height=False):
                    with gr.Column(scale=5):
                        image = gr.Image(type="pil", label="Optional reference image", height=220)
                        prompt = gr.Textbox(
                            label="Video description",
                            placeholder="Describe subject, action, environment, camera movement, lighting and visual style…",
                            lines=6,
                        )
                        negative = gr.Textbox(label="Avoid", value=NEGATIVE_PROMPT, lines=2)
                        with gr.Row():
                            resolution = gr.Dropdown(list(RESOLUTION_PRESETS), value=DEFAULT_RESOLUTION, label="Native generation size")
                            duration = gr.Dropdown(list(DURATION_PRESETS), value=DEFAULT_DURATION, label="Clip duration")
                        with gr.Row():
                            export_preset = gr.Dropdown(list(EXPORT_PRESETS), value="Native MP4", label="Download size")
                            seed = gr.Number(value=-1, precision=0, label="Seed (-1 random)")
                        with gr.Accordion("Advanced", open=False):
                            steps = gr.Slider(8, 40, value=DEFAULT_NUM_INFERENCE_STEPS, step=1, label="Inference steps")
                            guidance = gr.Slider(1.0, 7.0, value=DEFAULT_GUIDANCE_SCALE, step=0.1, label="Guidance")
                        generate = gr.Button("Generate video", variant="primary", elem_classes=["primary-action"])
                    with gr.Column(scale=6):
                        output = gr.Video(label="Generated video", autoplay=True, height=470)
                        status = gr.Textbox(label="Generation status", lines=8, interactive=False)
                generate.click(generate_single, [prompt, image, negative, resolution, duration, steps, guidance, seed, export_preset], [output, status])

            with gr.Tab("🧸 Cartoon Story Studio"):
                gr.Markdown("Build a long story from short LTX scenes. The final frame of each scene becomes the visual reference for the next scene, while the **character bible** is injected into every prompt.")
                with gr.Row(equal_height=False):
                    with gr.Column(scale=5):
                        story = gr.Textbox(
                            label="Story / scene beats",
                            placeholder="One line per scene works best.\nScene 1: Milo discovers a glowing moon-cookie…\nScene 2: The cookie rolls into a tiny rocket…",
                            lines=9,
                        )
                        character_bible = gr.Textbox(
                            label="Character bible — keep this exact across scenes",
                            placeholder="Milo: small orange fox, teal scarf, green eyes, rounded ears. Luna: tiny blue robot, yellow screen face…",
                            lines=4,
                        )
                        reference = gr.Image(type="pil", label="Optional character/style reference for Scene 1", height=210)
                        with gr.Row():
                            style = gr.Dropdown(list(CARTOON_STYLES), value="Premium 3D Kids Animation", label="Cartoon style")
                            scene_count = gr.Slider(1, MAX_STORY_SCENES, value=DEFAULT_STORY_SCENES, step=1, label="Scenes")
                        with gr.Row():
                            story_resolution = gr.Dropdown(list(RESOLUTION_PRESETS), value=DEFAULT_RESOLUTION, label="Scene size")
                            story_duration = gr.Dropdown(list(DURATION_PRESETS), value=DEFAULT_DURATION, label="Seconds per scene")
                        with gr.Row():
                            story_export = gr.Dropdown(list(EXPORT_PRESETS), value="Native MP4", label="Final download size")
                            story_seed = gr.Number(value=1001, precision=0, label="Base seed")
                        with gr.Accordion("Story generation advanced", open=False):
                            story_steps = gr.Slider(8, 40, value=DEFAULT_NUM_INFERENCE_STEPS, step=1, label="Inference steps")
                            story_guidance = gr.Slider(1.0, 7.0, value=DEFAULT_GUIDANCE_SCALE, step=0.1, label="Guidance")
                            story_negative = gr.Textbox(value=NEGATIVE_PROMPT, label="Avoid", lines=2)
                        with gr.Row():
                            preview = gr.Button("Preview storyboard", elem_classes=["soft-btn"])
                            story_generate = gr.Button("Generate full cartoon", variant="primary", elem_classes=["primary-action"])
                    with gr.Column(scale=6):
                        storyboard = gr.Markdown("### Your storyboard will appear here", elem_classes=["storyboard-card"])
                        story_output = gr.Video(label="Continuous cartoon", autoplay=True, height=390)
                        story_status = gr.Textbox(label="Story progress", lines=10, interactive=False)

                preview.click(preview_story, [story, scene_count, style, character_bible], storyboard)
                story_generate.click(
                    generate_story,
                    [story, character_bible, style, scene_count, reference, story_resolution, story_duration, story_steps, story_guidance, story_seed, story_negative, story_export],
                    [story_output, story_status],
                )

            with gr.Tab("⚙️ RTX 4050 Guide"):
                gr.Markdown("""
### Recommended settings for your Predator laptop

**Start here:** 384×224, 121 frames (~4.0 s), 20 steps, one scene at a time. If stable, try 512×288 or 161/193 frames. The 241-frame option is the practical single-clip ceiling exposed by this UI because LTX works best below 257 frames.

**Long cartoons:** do not try to create a one-minute clip in one pass on 6 GB VRAM. Use Cartoon Story Studio. It generates each short scene, extracts the final frame, conditions the next scene, and stitches the clips without loading the whole movie into RAM.

**1080p download:** the 1080p preset is a high-quality delivery resize. It does not invent native 1080p detail; generating native 1080p on this GPU is intentionally disabled to avoid CUDA OOM.

**Offline use:** `python run.py` installs missing Python packages and downloads the Hugging Face model the first time. After the model files are cached, generation can run without an API key.
                """)

        gr.HTML('<footer class="app-footer">Built for local creation • No MiniMax API • No cloud generation required after model download</footer>')
    return app


if __name__ == "__main__":
    create_app().launch(server_name="127.0.0.1", server_port=7860, inbrowser=True, show_error=True)
