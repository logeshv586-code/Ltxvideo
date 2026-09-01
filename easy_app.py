"""Simple customer-facing personal video maker."""
from __future__ import annotations

import gradio as gr

from config import NEGATIVE_PROMPT
from engine.longform import (
    ASPECT_LABELS,
    CLIP_LENGTHS,
    DURATION_SECONDS,
    GENERATION_MODES,
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
    "3D Animation": (
        "premium stylized 3D animated film, expressive appealing characters, clean rounded geometry, "
        "stable proportions, polished materials, soft cinematic global illumination, readable acting, "
        "coherent props and environment, rich but controlled color, crisp silhouettes"
    ),
    "Cinematic": (
        "cinematic realistic video, natural materials, believable motion, controlled film lighting, "
        "clean dynamic range and polished color grading"
    ),
    "Clay Animation": (
        "high-quality handcrafted clay animation, tactile clay textures, miniature sets, charming characters, "
        "soft studio lighting and smooth generated motion"
    ),
    "Anime": (
        "polished anime film look, expressive acting, detailed painted backgrounds, clean linework, cinematic composition"
    ),
    "Product Video": (
        "premium product commercial, accurate product geometry, crisp material detail, controlled reflections, "
        "studio lighting and deliberate camera movement"
    ),
}

DEFAULT_NEGATIVE = (
    "blurry, smeared textures, warped geometry, distorted face, deformed limbs, duplicate subjects, extra limbs, "
    "broken anatomy, inconsistent character design, identity drift, flicker, jitter, frozen frames, morphing objects, "
    "muddy lighting, low detail, text, watermark, logo"
)

CSS = """
.gradio-container { max-width: 1240px !important; }
.hero { padding: 26px 30px; border-radius: 22px; margin: 6px 0 18px;
        background: linear-gradient(135deg, rgba(79,70,229,.16), rgba(14,165,233,.09));
        border: 1px solid rgba(120,120,150,.20); }
.hero h1 { font-size: 38px; margin: 0 0 8px; letter-spacing: -.035em; }
.hero p { margin:0; font-size:16px; line-height:1.55; opacity:.82; max-width:850px; }
.creator-card, .result-card { border:1px solid rgba(120,120,150,.18); border-radius:18px; padding:16px; }
.generate-btn { min-height: 56px !important; font-weight: 800 !important; border-radius: 14px !important; }
.small-note { opacity:.72; font-size:12px; }
"""


def _delivery_dimensions(aspect: str) -> tuple[int, int]:
    if aspect == "9:16":
        return 720, 1280
    if aspect == "1:1":
        return 720, 720
    return 1280, 720


def _mode_visibility(mode: str):
    single = mode == "Single Clip"
    return gr.update(visible=single), gr.update(visible=not single)


def _build_plan(
    description: str,
    mode: str,
    total_duration: str,
    clip_length: str,
    quality: str,
    aspect_label: str,
):
    return plan_story(
        description,
        total_duration,
        quality,
        aspect_label,
        generation_mode=mode,
        clip_length_label=clip_length,
    )


def preview_plan(description, mode, total_duration, clip_length, quality, aspect_label):
    try:
        return plan_markdown(_build_plan(description, mode, total_duration, clip_length, quality, aspect_label))
    except Exception as exc:
        return f"### Video setup\n\n{exc}"


def generate_video(
    description: str,
    mode: str,
    total_duration: str,
    clip_length: str,
    quality: str,
    aspect_label: str,
    style_name: str,
    reference_image,
    consistency: str,
    negative_prompt: str,
    seed: int,
    progress=gr.Progress(),
):
    if not description or not description.strip():
        return None, "### Video setup\n\nDescribe the video first.", "Please describe what you want to create."

    try:
        plan = _build_plan(description, mode, total_duration, clip_length, quality, aspect_label)
    except Exception as exc:
        return None, "### Video setup\n\nPlanning failed.", f"Planning failed: {exc}"

    logs = [
        f"Mode: {plan.generation_mode}",
        f"Requested duration: {plan.target_seconds}s",
        f"GPU clip size: {plan.clip_seconds:.1f}s × {plan.scene_count}",
        f"Quality: {plan.profile.label}",
    ]

    def callback(message: str, value: float) -> None:
        logs.append(message)
        progress(min(1.0, max(0.0, float(value))), desc=message)

    try:
        raw = LONGFORM.generate(
            plan=plan,
            style_prompt=STYLE_PRESETS[style_name],
            character_lock=consistency or "",
            reference_image=reference_image,
            negative_prompt=negative_prompt or DEFAULT_NEGATIVE or NEGATIVE_PROMPT,
            seed=int(seed),
            progress_callback=callback,
        )

        out_w, out_h = _delivery_dimensions(plan.aspect)
        delivered = raw.with_name(raw.stem + f"_{out_w}x{out_h}.mp4")
        progress(0.998, desc="Enhancing final video")
        export_delivery(
            raw,
            delivered,
            out_w,
            out_h,
            enhance_quality=True,
            target_fps=plan.profile.fps,
            duration_seconds=plan.target_seconds,
        )
        logs.append(f"Done: {delivered.name}")
        return str(delivered), plan_markdown(plan), "\n".join(logs[-40:])
    except Exception as exc:
        # Keep the page usable and show the real backend problem in the status
        # box instead of only a generic Gradio red error badge.
        message = f"Generation stopped: {type(exc).__name__}: {exc}"
        logs.append(message)
        return None, plan_markdown(plan), "\n".join(logs[-40:])


def create_app() -> gr.Blocks:
    theme = gr.themes.Soft(
        primary_hue=gr.themes.colors.indigo,
        secondary_hue=gr.themes.colors.sky,
        neutral_hue=gr.themes.colors.slate,
    )

    with gr.Blocks(
        title="LTX Personal Video Maker",
        theme=theme,
        css=CSS,
        analytics_enabled=False,
    ) as app:
        gr.HTML(
            """
            <section class="hero">
              <h1>LTX Personal Video Maker</h1>
              <p>Describe the video once. Choose how long it should be, the quality and the output size.
              A 6 GB RTX 4050 renders safe 4-second or heavier 8-second clips; longer videos continue automatically
              part by part and are combined into one final video.</p>
            </section>
            """
        )

        with gr.Row(equal_height=False):
            with gr.Column(scale=6, elem_classes=["creator-card"]):
                description = gr.Textbox(
                    label="Describe your video",
                    placeholder=(
                        "Example: On the moon, an orange fox meets a cow and a sheep. The fox helps them make a cookie. "
                        "After that they walk together across the moon while Earth glows in the background."
                    ),
                    lines=9,
                )

                mode = gr.Radio(
                    choices=list(GENERATION_MODES),
                    value="Continuous Video",
                    label="Video type",
                )

                with gr.Row():
                    clip_length = gr.Radio(
                        choices=list(CLIP_LENGTHS),
                        value="4 seconds • Recommended",
                        label="Single / backend clip length",
                        info="4 sec is safest for high quality. 8 sec is heavier but supported by the 4050 path.",
                        visible=False,
                    )
                    total_duration = gr.Dropdown(
                        choices=list(DURATION_SECONDS),
                        value="15 seconds",
                        label="Video duration",
                        info="The backend automatically creates as many 4s/8s parts as needed.",
                    )

                with gr.Row():
                    quality = gr.Radio(
                        choices=list(QUALITY_PROFILES),
                        value="Balanced",
                        label="Quality",
                    )
                    aspect = gr.Dropdown(
                        choices=list(ASPECT_LABELS),
                        value="Landscape • 1280×720 • 16:9",
                        label="Size / pixels",
                    )

                style = gr.Dropdown(
                    choices=list(STYLE_PRESETS),
                    value="3D Animation",
                    label="Look",
                )

                with gr.Accordion("Optional reference & consistency", open=False):
                    reference_image = gr.Image(
                        type="pil",
                        label="Reference image",
                        height=210,
                    )
                    consistency = gr.Textbox(
                        label="Keep this subject/character consistent",
                        placeholder="Example: Orange fox, green eyes, cream chest, no clothes; keep the same design in every part.",
                        lines=2,
                    )

                with gr.Accordion("Advanced", open=False):
                    negative = gr.Textbox(label="Avoid", value=DEFAULT_NEGATIVE, lines=3)
                    seed = gr.Number(value=-1, precision=0, label="Seed (-1 = random)")
                    hardware = gr.Markdown(get_status_markdown())
                    refresh = gr.Button("Refresh GPU status", size="sm")
                    refresh.click(get_status_markdown, outputs=hardware)

                with gr.Row():
                    preview = gr.Button("Preview")
                    generate = gr.Button("Generate video", variant="primary", elem_classes=["generate-btn"])

            with gr.Column(scale=5, elem_classes=["result-card"]):
                output = gr.Video(label="Your video", autoplay=True, height=430)
                plan_view = gr.Markdown(
                    "### Video setup\n\nChoose your settings and click **Preview** or **Generate video**."
                )
                status = gr.Textbox(label="Generation status", lines=12, interactive=False)

        mode.change(_mode_visibility, inputs=mode, outputs=[clip_length, total_duration])
        preview.click(
            preview_plan,
            [description, mode, total_duration, clip_length, quality, aspect],
            plan_view,
        )
        generate.click(
            generate_video,
            [
                description,
                mode,
                total_duration,
                clip_length,
                quality,
                aspect,
                style,
                reference_image,
                consistency,
                negative,
                seed,
            ],
            [output, plan_view, status],
        )

    return app
