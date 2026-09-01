"""Customer-first LTX Video Creator.

This is the default UI for normal users. Technical frame counts, inference
steps and GPU memory controls are selected automatically from three simple
quality modes. The original app.py remains available as the legacy advanced
studio for developers.
"""
from __future__ import annotations

from pathlib import Path

import gradio as gr

from config import NEGATIVE_PROMPT
from engine.longform import (
    ASPECT_LABELS,
    DURATION_SECONDS,
    QUALITY_PROFILES,
    LongFormVideoGenerator,
    plan_markdown,
    plan_story,
)
from engine.memory_manager import get_status_markdown
from engine.optimized_generator import OptimizedVideoGenerator
from engine.video_processor import export_delivery

GENERATOR = OptimizedVideoGenerator()
LONGFORM = LongFormVideoGenerator(GENERATOR)

STYLE_PRESETS = {
    "Cinematic Real World": (
        "photorealistic cinematic footage, natural materials and skin, physically believable motion, "
        "clean dynamic range, controlled film lighting, polished but realistic color grading"
    ),
    "Premium 3D Animation": (
        "premium stylized 3D animation, expressive characters, clean geometry, detailed materials, "
        "soft global illumination, cinematic family-film quality"
    ),
    "Clay Animation": (
        "high-quality handcrafted clay animation, tactile clay textures, miniature practical sets, "
        "soft studio lighting, charming stop-motion character design with smooth generated motion"
    ),
    "Anime Film": (
        "polished anime feature-film look, expressive acting, detailed painted backgrounds, "
        "clean linework, cinematic composition and controlled motion"
    ),
    "Product Commercial": (
        "premium product advertising cinematography, accurate product geometry, crisp material detail, "
        "controlled reflections, studio lighting and deliberate camera movement"
    ),
    "Travel Film": (
        "high-end travel cinematography, atmospheric depth, authentic environmental textures, "
        "natural people and movement, elegant camera language"
    ),
    "Graphic Novel": (
        "high-end animated graphic novel, confident ink contours, textured shadows, rich controlled color, "
        "cinematic framing and subtle dimensional parallax"
    ),
}

CSS = """
.gradio-container { max-width: 1280px !important; }
.hero { padding: 30px 34px; border-radius: 24px; margin: 4px 0 18px;
        background: linear-gradient(135deg, rgba(99,102,241,.13), rgba(6,182,212,.10));
        border: 1px solid rgba(120,120,150,.20); }
.hero h1 { font-size: 40px; margin: 0 0 8px; letter-spacing: -.035em; }
.hero p { font-size: 16px; max-width: 900px; line-height: 1.6; opacity: .82; }
.hero .chips { display:flex; gap:8px; flex-wrap:wrap; margin-top:14px; }
.hero .chips span { padding:6px 10px; border-radius:999px; border:1px solid rgba(120,120,150,.22); font-size:12px; }
.primary-create { min-height: 54px !important; font-weight: 800 !important; border-radius: 14px !important; }
.plan-card { border:1px solid rgba(120,120,150,.18); border-radius:16px; padding:14px 18px; }
.help-card { padding: 14px 18px; border-radius: 16px; background: rgba(120,120,150,.06); }
"""


def _progress_bridge(progress, logs: list[str]):
    def callback(message: str, value: float) -> None:
        logs.append(message)
        progress(min(1.0, max(0.0, float(value))), desc=message)
    return callback


def _delivery_dimensions(aspect: str, quality: str) -> tuple[int, int]:
    long_edge = QUALITY_PROFILES[quality].delivery_long_edge
    if aspect == "9:16":
        return (720, 1280) if long_edge <= 1280 else (1080, 1920)
    if aspect == "1:1":
        edge = 720 if long_edge <= 1280 else 1080
        return edge, edge
    return (1280, 720) if long_edge <= 1280 else (1920, 1080)


def preview_plan(story: str, duration: str, quality: str, aspect_label: str):
    try:
        plan = plan_story(story, duration, quality, aspect_label)
        return plan_markdown(plan)
    except ValueError as exc:
        return f"### Automatic video plan\n\n{exc}"


def generate_video(
    story: str,
    reference_image,
    character_lock: str,
    style_name: str,
    duration: str,
    quality: str,
    aspect_label: str,
    negative_prompt: str,
    seed: int,
    progress=gr.Progress(),
):
    if not story or not story.strip():
        raise gr.Error("Describe the video or paste your story first.")

    try:
        plan = plan_story(story, duration, quality, aspect_label)
    except ValueError as exc:
        raise gr.Error(str(exc)) from exc

    style_prompt = STYLE_PRESETS[style_name]
    logs: list[str] = [
        f"Automatic plan: {plan.scene_count} scenes · ~{plan.estimated_seconds:.0f}s · {quality} mode",
        f"Native scene render: {plan.width}x{plan.height}; final delivery is upscaled after generation.",
    ]
    callback = _progress_bridge(progress, logs)

    raw_path = LONGFORM.generate(
        plan=plan,
        style_prompt=style_prompt,
        character_lock=character_lock or "",
        reference_image=reference_image,
        negative_prompt=negative_prompt or NEGATIVE_PROMPT,
        seed=int(seed),
        progress_callback=callback,
    )

    delivery_width, delivery_height = _delivery_dimensions(plan.aspect, quality)
    delivered = raw_path.with_name(raw_path.stem + f"_{delivery_width}x{delivery_height}.mp4")
    progress(0.998, desc="Preparing social-media delivery file")
    export_delivery(
        raw_path,
        delivered,
        delivery_width,
        delivery_height,
        enhance_quality=True,
    )
    logs.append(f"Complete: {delivered.name}")
    logs.append(
        "Note: delivery resolution is a high-quality upscale of the native LTX render; "
        "native detail is controlled by the selected quality mode."
    )
    return str(delivered), plan_markdown(plan), "\n".join(logs[-30:])


def create_app() -> gr.Blocks:
    theme = gr.themes.Soft(
        primary_hue=gr.themes.colors.indigo,
        secondary_hue=gr.themes.colors.cyan,
        neutral_hue=gr.themes.colors.slate,
    )

    with gr.Blocks(title="LTX Easy Video Creator", theme=theme, css=CSS) as app:
        gr.HTML(
            """
            <section class="hero">
              <h1>LTX Easy Video Creator</h1>
              <p>Paste one idea, paragraph, script or story. The creator automatically decides the scene count,
              generates every scene in order, carries visual continuity forward, and assembles one final video.
              No frame-count or diffusion knowledge is required.</p>
              <div class="chips">
                <span>RTX 4050 6 GB optimized</span><span>Auto scene planning</span><span>Up to 5 minutes</span>
                <span>YouTube 16:9</span><span>Reels / Shorts 9:16</span><span>Sequential low-VRAM rendering</span>
              </div>
            </section>
            """
        )

        with gr.Row():
            hardware = gr.Markdown(get_status_markdown(), elem_classes=["help-card"])
            refresh = gr.Button("Refresh GPU status")
            refresh.click(get_status_markdown, outputs=hardware)

        with gr.Row(equal_height=False):
            with gr.Column(scale=6):
                story = gr.Textbox(
                    label="1. Describe your complete video",
                    placeholder=(
                        "Example: A small clay astronaut wakes inside a moon base, notices a glowing cookie on the table, "
                        "walks toward it, picks it up, then the room lights flicker and a tiny alien appears...\n\n"
                        "You can paste a full paragraph or a long script."
                    ),
                    lines=12,
                )
                reference_image = gr.Image(
                    type="pil",
                    label="Optional starting image / character reference",
                    height=220,
                )
                character_lock = gr.Textbox(
                    label="Optional character consistency",
                    placeholder="Example: Milo is an orange fox with green eyes, teal scarf and brown boots. Keep this exact design throughout.",
                    lines=2,
                )

                with gr.Row():
                    style = gr.Dropdown(
                        choices=list(STYLE_PRESETS),
                        value="Cinematic Real World",
                        label="2. Visual style",
                    )
                    aspect = gr.Dropdown(
                        choices=list(ASPECT_LABELS),
                        value="YouTube / Landscape (16:9)",
                        label="3. Where will you publish?",
                    )

                with gr.Row():
                    duration = gr.Dropdown(
                        choices=list(DURATION_SECONDS),
                        value="Auto from story",
                        label="4. Video length",
                        info="Auto estimates length from your paragraph. You can also force 15 sec to 5 min.",
                    )
                    quality = gr.Radio(
                        choices=list(QUALITY_PROFILES),
                        value="Balanced",
                        label="5. Quality / speed",
                        info="Balanced is recommended for RTX 4050 6 GB.",
                    )

                with gr.Accordion("Advanced (usually leave unchanged)", open=False):
                    negative = gr.Textbox(label="Avoid", value=NEGATIVE_PROMPT, lines=3)
                    seed = gr.Number(value=-1, precision=0, label="Seed (-1 = random)")

                with gr.Row():
                    preview = gr.Button("Preview automatic scene plan")
                    generate = gr.Button("Create complete video", variant="primary", elem_classes=["primary-create"])

            with gr.Column(scale=5):
                plan_view = gr.Markdown(
                    "### Automatic video plan\n\nEnter your story and click **Preview automatic scene plan**.",
                    elem_classes=["plan-card"],
                )
                output = gr.Video(label="Final video", autoplay=True, height=430)
                status = gr.Textbox(label="Progress", lines=10, interactive=False)

        gr.Markdown(
            "**For developers:** the older multi-studio interface is still kept in `app.py`. "
            "The default launcher now uses this simplified customer interface."
        )

        preview.click(preview_plan, [story, duration, quality, aspect], plan_view)
        generate.click(
            generate_video,
            [story, reference_image, character_lock, style, duration, quality, aspect, negative, seed],
            [output, plan_view, status],
        )

    return app
