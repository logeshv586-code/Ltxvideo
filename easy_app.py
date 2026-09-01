"""Simple customer-facing personal video maker."""
from __future__ import annotations

import re

import gradio as gr

from config import NEGATIVE_PROMPT
from engine.longform import (
    ASPECT_LABELS,
    CLIP_LENGTHS,
    CUSTOMER_QUALITY_CHOICES,
    DURATION_SECONDS,
    GENERATION_MODES,
    LongFormVideoGenerator,
    plan_markdown,
    plan_story,
)
from engine.memory_manager import get_status_markdown
from engine.optimized_generator import OptimizedVideoGenerator
from engine.video_processor import export_delivery

GENERATOR = OptimizedVideoGenerator()
LONGFORM = LongFormVideoGenerator(GENERATOR)

AUTO_LOOK = "Auto • detect from description"
STYLE_PRESETS = {
    "3D Animation": (
        "premium stylized 3D animated film, expressive appealing characters, clean rounded geometry, stable proportions, "
        "polished materials with visible texture detail, soft cinematic global illumination, readable facial acting, "
        "coherent props and environment, rich controlled color, crisp silhouettes, family-feature animation finish"
    ),
    "Cinematic": (
        "cinematic realistic video, natural materials, believable motion, controlled film lighting, clean dynamic range, "
        "fine surface texture, realistic depth and polished color grading"
    ),
    "Clay Animation": (
        "premium handcrafted clay animation, clearly visible tactile clay texture, miniature practical-set look, charming rounded characters, "
        "soft cinematic studio lighting, expressive stop-motion-style acting, clean facial features and polished family-film finish"
    ),
    "Anime": (
        "polished anime feature-film look, expressive acting, detailed painted backgrounds, clean linework, controlled cel shading, "
        "cinematic composition and stable character design"
    ),
    "Product Video": (
        "premium product commercial, accurate product geometry, crisp material detail, controlled reflections, studio lighting, "
        "clean edges and deliberate camera movement"
    ),
}
LOOK_CHOICES = [AUTO_LOOK, *STYLE_PRESETS.keys()]

DEFAULT_NEGATIVE = (
    "blurry, smeared texture, muddy detail, warped geometry, distorted face, deformed limbs, duplicate subjects, extra limbs, "
    "broken anatomy, inconsistent character design, identity drift, flicker, jitter, frozen frames, morphing objects, "
    "low detail, melted features, unreadable text, subtitles, watermark, logo"
)

CSS = """
.gradio-container { max-width: 1240px !important; }
.hero { padding: 24px 28px; border-radius: 22px; margin: 6px 0 18px;
        background: linear-gradient(135deg, rgba(79,70,229,.16), rgba(14,165,233,.09));
        border: 1px solid rgba(120,120,150,.20); }
.hero h1 { font-size: 38px; margin: 0 0 8px; letter-spacing: -.035em; }
.hero p { margin:0; font-size:16px; line-height:1.55; opacity:.82; max-width:900px; }
.creator-card, .result-card { border:1px solid rgba(120,120,150,.18); border-radius:18px; padding:16px; }
.generate-btn { min-height: 56px !important; font-weight: 800 !important; border-radius: 14px !important; }
.quality-note { padding:10px 12px; border-radius:12px; background:rgba(99,102,241,.08); font-size:12px; line-height:1.45; }
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


def _detect_look(description: str, style_hint: str) -> str:
    text = f"{style_hint} {description}".lower()
    if re.search(r"\b(clay|claymation|stop[ -]?motion|handcrafted clay)\b", text):
        return "Clay Animation"
    if re.search(r"\b(anime|manga|cel[ -]?shaded)\b", text):
        return "Anime"
    if re.search(r"\b(product|commercial|advert|packshot)\b", text):
        return "Product Video"
    if re.search(r"\b(realistic|photoreal|live[ -]?action|cinematic real)\b", text):
        return "Cinematic"
    if re.search(r"\b(3d|cartoon|kids animation|animated character|pixar|family animation)\b", text):
        return "3D Animation"
    # The current product is mainly being tested as an animated personal video
    # maker, so a clean 3D look is the least surprising automatic fallback.
    return "3D Animation"


def _resolved_look(selected: str, description: str, style_hint: str) -> tuple[str, str]:
    name = _detect_look(description, style_hint) if selected == AUTO_LOOK else selected
    return name, STYLE_PRESETS.get(name, STYLE_PRESETS["3D Animation"])


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

    actual_look, style_prompt = _resolved_look(style_name, description, plan.style_hint)
    logs = [
        f"Mode: {plan.generation_mode}",
        f"Requested duration: {plan.target_seconds}s",
        f"GPU clip size: {plan.clip_seconds:.1f}s × {plan.scene_count}",
        f"Native render: {plan.width}x{plan.height} @ {plan.profile.fps} fps",
        f"Inference: {plan.profile.inference_steps} steps · guidance {plan.profile.guidance_scale}",
        f"Look: {actual_look}",
    ]

    def callback(message: str, value: float) -> None:
        logs.append(message)
        progress(min(1.0, max(0.0, float(value))), desc=message)

    try:
        raw = LONGFORM.generate(
            plan=plan,
            style_prompt=style_prompt,
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
        return str(delivered), plan_markdown(plan), "\n".join(logs[-50:])
    except Exception as exc:
        message = f"Generation stopped: {type(exc).__name__}: {exc}"
        logs.append(message)
        return None, plan_markdown(plan), "\n".join(logs[-50:])


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
              <p>Describe what should happen. Choose single or continuous video, duration, quality and output size.
              The app focuses each GPU render on one achievable action, continues longer videos part-by-part, then prepares a clean 720p delivery file.</p>
            </section>
            """
        )

        with gr.Row(equal_height=False):
            with gr.Column(scale=6, elem_classes=["creator-card"]):
                description = gr.Textbox(
                    label="Describe your video",
                    placeholder=(
                        "Example: On the moon, an orange fox meets a cow and a sheep. The fox helps them make a cookie. "
                        "After that they walk together while Earth glows in the background.\n\n"
                        "You can also add lines such as Style: clay animation or Camera: slow zoom in."
                    ),
                    lines=10,
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
                        info="4 sec gives the best detail. High + 8 sec automatically uses a memory-safe native size.",
                        visible=False,
                    )
                    total_duration = gr.Dropdown(
                        choices=list(DURATION_SECONDS),
                        value="15 seconds",
                        label="Video duration",
                        info="Long videos are generated in safe GPU-sized parts and combined automatically.",
                    )

                with gr.Row():
                    quality = gr.Radio(
                        choices=list(CUSTOMER_QUALITY_CHOICES),
                        value="High",
                        label="Quality",
                        info="High is the default for final output. Balanced is faster and safer for long runs.",
                    )
                    aspect = gr.Dropdown(
                        choices=list(ASPECT_LABELS),
                        value="Landscape • 1280×720 • 16:9",
                        label="Size / pixels",
                    )

                gr.HTML(
                    "<div class='quality-note'><b>Final-quality policy:</b> the old 384×224 Fast mode is now draft/internal only. "
                    "High 4-second clips generate at 576×320 natively before 720p delivery; 8-second High clips automatically use a 512×288 memory-safe native render.</div>"
                )

                style = gr.Dropdown(
                    choices=LOOK_CHOICES,
                    value=AUTO_LOOK,
                    label="Look",
                    info="Auto reads explicit Style/Look instructions from your description so the UI does not conflict with your prompt.",
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
                status = gr.Textbox(label="Generation status", lines=14, interactive=False)

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