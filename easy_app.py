"""Simple customer-facing personal video maker."""
from __future__ import annotations

import re
from queue import Empty, Queue
from threading import Thread
from time import monotonic

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
:root {
  --canvas: #090d16; --surface: #111827; --surface-raised: #151d2c;
  --field: #0d1421; --line: #2a3548; --line-soft: #202a3a;
  --ink: #f8fafc; --muted: #9aa8bc; --accent: #8b5cf6; --accent-strong: #7c3aed;
}
body, .gradio-container { background: var(--canvas) !important; color: var(--ink) !important; }
.gradio-container {
  max-width: 1480px !important; padding: 24px 28px 44px !important;
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif !important;
}
.studio-header { display:flex; align-items:center; gap:28px; padding:0 4px 22px; border-bottom:1px solid var(--line-soft); margin-bottom:22px; }
.brand-mark { font-size:27px; font-weight:850; letter-spacing:-.07em; color:#fff; padding-right:28px; border-right:1px solid #344155; }
.brand-name { font-size:20px; font-weight:720; letter-spacing:-.035em; white-space:nowrap; }
.steps { display:flex; align-items:center; gap:12px; margin-left:auto; color:var(--muted); font-size:14px; font-weight:650; }
.step { display:flex; align-items:center; gap:8px; white-space:nowrap; }.step b { display:grid; place-items:center; width:29px; height:29px; border:1px solid #64748b; border-radius:50%; font-size:13px; }
.step.active { color:#c4b5fd; }.step.active b { border-color:var(--accent); color:#d8b4fe; background:#211542; }
.step-line { width:48px; height:1px; background:#3a4557; }
.workspace { align-items:stretch !important; gap:18px !important; }
.creator-card, .result-card { border:1px solid var(--line); border-radius:14px; background:var(--surface); box-shadow:0 22px 50px rgba(0,0,0,.16); }
.creator-card { padding:20px !important; }.result-card { padding:18px !important; position:sticky; top:12px; height:fit-content; }
.section-heading { margin:4px 0 13px; color:#fff; font-size:16px; font-weight:720; letter-spacing:-.02em; }.section-heading.compact { margin-top:22px; }
.section-help { color:var(--muted); font-size:13px; margin:-7px 0 13px; }
.prompt-wrap { padding:0; border:0; background:transparent; }
label, .gradio-container .block-label { color:#dce4f0 !important; font-size:13px !important; font-weight:650 !important; letter-spacing:0 !important; }
.gradio-container .block-label { background:transparent !important; }
.gradio-container .block-label span, .gradio-container .block-label > div { background:transparent !important; color:#cbd5e1 !important; }
.gradio-container .block, .gradio-container .block > .wrap, .gradio-container .block > .form, .gradio-container .form, .gradio-container fieldset, .gradio-container .wrap, .gradio-container textarea, .gradio-container input { border-color:var(--line) !important; background:var(--field) !important; color:var(--ink) !important; box-shadow:none !important; }
.gradio-container textarea::placeholder, .gradio-container input::placeholder { color:#708098 !important; }
.gradio-container .wrap:focus-within { border-color:#7c5ccc !important; box-shadow:0 0 0 3px rgba(139,92,246,.13) !important; }
.gradio-container .prose, .gradio-container .prose * { color:var(--muted) !important; }
.gradio-container .block { border-color:var(--line) !important; }
.choice-row { gap:10px !important; }.choice-row > .form { min-height:118px; padding:14px !important; border-radius:11px !important; background:var(--field) !important; }
.style-picker { margin-top:0; }.style-picker fieldset { gap:8px !important; }.style-picker label { min-width:0 !important; }
.gradio-container input[type="radio"] + span { color:#d8e0ec !important; }
.gradio-container label:has(input[type="radio"]) { background:var(--field) !important; border-color:var(--line) !important; }
.gradio-container label:has(input[type="radio"]:checked) { border-color:#8b5cf6 !important; background:rgba(124,58,237,.13) !important; }
.gradio-container button { min-height:48px !important; border-radius:10px !important; font-size:15px !important; font-weight:700 !important; background:var(--field) !important; border-color:#506077 !important; color:#f1f5f9 !important; transition:transform .16s ease, background .16s ease !important; }
.gradio-container button:hover { transform:translateY(-1px); }.generate-btn button, button.primary { background:var(--accent-strong) !important; border-color:#8b5cf6 !important; color:#fff !important; box-shadow:0 10px 24px rgba(109,40,217,.24) !important; }
.preview-btn button { background:transparent !important; border-color:#506077 !important; color:#f1f5f9 !important; }
.quality-note { margin-top:15px; padding:10px 12px; border-left:2px solid var(--accent); border-radius:7px; background:#171b30; color:#c9d4e5; font-size:13px; line-height:1.45; }
.result-title { margin:1px 0 13px; font-size:18px; font-weight:720; color:#fff; letter-spacing:-.03em; }.result-subtitle { margin:-8px 0 14px; font-size:13px; color:var(--muted); }
.result-card video { border-radius:10px !important; background:#070b12 !important; border:1px solid var(--line-soft) !important; }
.result-card .wrap { background:#0b101a !important; }.status-box textarea { min-height:240px !important; font-family:Inter, ui-sans-serif, sans-serif !important; font-size:13px !important; line-height:1.65 !important; color:#bdc8d9 !important; }
.plan-box { margin-top:14px; border-top:1px solid var(--line-soft); padding-top:12px; background:var(--field) !important; }.plan-box > *, .plan-box .prose { background:var(--field) !important; }.plan-box h3 { color:#e5e7eb !important; font-size:14px !important; }
.gradio-container .accordion, .gradio-container .accordion > *, .gradio-container .accordion button { background:var(--field) !important; border-color:var(--line) !important; }
.gradio-container .accordion button, .gradio-container .accordion button * { color:#dce4f0 !important; }
.gradio-container .accordion { border:1px solid var(--line) !important; border-radius:9px !important; margin-top:9px !important; }
.gradio-container .accordion summary { color:#e4eaf4 !important; font-weight:650 !important; }
@media (max-width: 900px) { .gradio-container { padding:16px !important; }.studio-header { gap:15px; align-items:flex-start; }.brand-mark { padding-right:15px; }.brand-name { font-size:17px; }.steps { display:none; }.result-card { position:static; }.choice-row > .form { min-height:auto; } }
"""

THEME = gr.themes.Soft(
    primary_hue=gr.themes.colors.indigo,
    secondary_hue=gr.themes.colors.sky,
    neutral_hue=gr.themes.colors.slate,
)


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
    dialogue_audio: str,
    voice_mode: str,
    voice_language: str,
    reference_image,
    consistency: str,
    negative_prompt: str,
    seed: int | float,
):
    """Stream backend generation status so a long T4 render never looks frozen."""
    if not description or not description.strip():
        yield None, "### Video setup\n\nDescribe the video first.", "Please describe what you want to create."
        return

    try:
        plan = _build_plan(description, mode, total_duration, clip_length, quality, aspect_label)
    except Exception as exc:
        yield None, "### Video setup\n\nPlanning failed.", f"Planning failed: {exc}"
        return

    actual_look, style_prompt = _resolved_look(style_name, description, plan.style_hint)
    logs = [
        f"Mode: {plan.generation_mode}",
        f"Requested duration: {plan.target_seconds}s",
        f"GPU clip size: {plan.clip_seconds:.1f}s x {plan.scene_count}",
        f"Native render: {plan.width}x{plan.height} @ {plan.profile.fps} fps",
        f"Inference: {plan.profile.inference_steps} steps, guidance {plan.profile.guidance_scale}",
        f"Look: {actual_look}",
    ]
    plan_view = plan_markdown(plan)
    events: Queue[tuple[str, float]] = Queue()
    result: dict[str, str | None] = {"video": None, "error": None}

    def callback(message: str, value: float) -> None:
        events.put((message, min(1.0, max(0.0, float(value)))))

    def work() -> None:
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
            callback("Preparing download-quality MP4…", 0.998)
            export_delivery(
                raw,
                delivered,
                out_w,
                out_h,
                enhance_quality=True,
                target_fps=plan.profile.fps,
                duration_seconds=plan.target_seconds,
            )

            from engine.audio_processor import generate_speech, resolve_narration

            narration, narration_language = resolve_narration(
                prompt=description,
                custom_text=dialogue_audio,
                mode=voice_mode,
                requested_language=voice_language,
            )
            if narration:
                callback(f"Generating {narration_language} narration…", 0.999)
                try:
                    from engine.audio_processor import add_audio_to_video

                    audio_path = delivered.with_suffix(".mp3")
                    final_path = delivered.with_name(delivered.stem + "_audio.mp4")
                    generate_speech(narration, audio_path, language=narration_language)
                    add_audio_to_video(delivered, audio_path, final_path)
                    delivered = final_path
                    callback(f"{narration_language} narration attached.", 0.999)
                except Exception as exc:
                    callback(f"Narration skipped: {type(exc).__name__}: {exc}", 0.999)

            result["video"] = str(delivered)
            events.put((f"Done: {delivered.name}", 1.0))
        except Exception as exc:
            result["error"] = f"Generation stopped: {type(exc).__name__}: {exc}"
            events.put((result["error"], 1.0))

    worker = Thread(target=work, name="ltx-video-generation", daemon=True)
    worker.start()
    start = monotonic()
    last_update = "Waiting for the GPU worker…"
    yield None, plan_view, "\n".join(logs + [last_update])

    while worker.is_alive() or not events.empty():
        try:
            message, value = events.get(timeout=2)
            last_update = f"{value * 100:.0f}% · {message}"
            logs.append(last_update)
            yield None, plan_view, "\n".join(logs[-18:])
        except Empty:
            elapsed = int(monotonic() - start)
            yield None, plan_view, "\n".join(logs[-17:] + [f"Still working ({elapsed}s) · {last_update}"])

    if result["error"]:
        yield None, plan_view, "\n".join(logs[-18:])
        return

    yield result["video"], plan_view, "\n".join(logs[-18:])


def create_app() -> gr.Blocks:
    with gr.Blocks(
        title="LTX Personal Video Maker",
        analytics_enabled=False,
    ) as app:
        gr.HTML("""
            <header class="studio-header">
              <div class="brand-mark">LTX</div><div class="brand-name">Easy Video Creator</div>
              <div class="steps" aria-label="Creation steps">
                <span class="step active"><b>1</b> Describe</span><i class="step-line"></i>
                <span class="step"><b>2</b> Choose</span><i class="step-line"></i><span class="step"><b>3</b> Create</span>
              </div>
            </header>
        """)

        with gr.Row(equal_height=False, elem_classes=["workspace"]):
            with gr.Column(scale=6, elem_classes=["creator-card"]):
                gr.HTML("<div class='section-heading'>Describe your scene</div><div class='section-help'>Include the subject, action, setting, and camera movement. You can add a style in your description too.</div>")
                description = gr.Textbox(
                    label="Describe your video",
                    placeholder=(
                        "Example: On the moon, an orange fox meets a cow and a sheep. The fox helps them make a cookie. "
                        "After that they walk together while Earth glows in the background.\n\n"
                        "You can also add lines such as Style: clay animation or Camera: slow zoom in."
                    ),
                    lines=7,
                )

                gr.HTML("<div class='section-heading compact'>Choose your setup</div><div class='section-help'>Start simple. Open the optional controls only when you need them.</div>")
                with gr.Row(elem_classes=["choice-row"]):
                    mode = gr.Radio(choices=list(GENERATION_MODES), value="Single Clip", label="Video type")
                    clip_length = gr.Radio(
                        choices=list(CLIP_LENGTHS),
                        value="4 seconds \u2022 Recommended",
                        label="Clip length",
                        info="Four seconds gives the best detail.",
                        visible=True,
                    )
                    total_duration = gr.Dropdown(
                        choices=list(DURATION_SECONDS),
                        value="15 seconds",
                        label="Duration",
                        info="Long videos are assembled safely.",
                    )
                    quality = gr.Radio(
                        choices=list(CUSTOMER_QUALITY_CHOICES),
                        value="Balanced",
                        label="Quality",
                        info="Balanced is quick and reliable.",
                    )
                    aspect = gr.Dropdown(
                        choices=list(ASPECT_LABELS),
                        value="Landscape \u2022 1280\u00d7720 \u2022 16:9",
                        label="Format",
                    )

                gr.HTML(
                    "<div class='quality-note'><b>Recommended first video:</b> Single Clip · 4 seconds · Balanced. "
                    "It is the fastest reliable way to confirm your T4 is working before creating a longer video.</div>"
                )

                gr.HTML("<div class='section-heading compact'>Visual style</div><div class='section-help'>Pick a look, or let the description decide automatically.</div>")
                style = gr.Radio(
                    choices=LOOK_CHOICES,
                    value=AUTO_LOOK,
                    label="Visual style",
                    elem_classes=["style-picker"],
                )

                with gr.Accordion("Voice & narration", open=False):
                    voice_mode = gr.Radio(
                        choices=("Automatic from prompt", "Custom narration", "No voice"),
                        value="Automatic from prompt",
                        label="Narration",
                        info="Automatic uses the first sentence, or an explicit Narration: line in your prompt.",
                    )
                    voice_language = gr.Dropdown(
                        choices=("Auto detect", "Tamil", "English"),
                        value="Auto detect",
                        label="Language",
                        info="Tamil script uses a Tamil neural voice; other text uses English.",
                    )
                    dialogue_audio = gr.Textbox(
                        label="Custom narration (optional)",
                        placeholder="Example: வணக்கம்! இது ஒரு அழகான கதை. Or: Welcome to a beautiful new story.",
                        lines=2,
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

                with gr.Accordion("Advanced settings", open=False):
                    negative = gr.Textbox(label="Avoid", value=DEFAULT_NEGATIVE, lines=3)
                    seed = gr.Number(value=-1, precision=0, label="Seed (-1 = random)")
                    hardware = gr.Markdown(get_status_markdown())
                    refresh = gr.Button("Refresh GPU status", size="sm")
                    refresh.click(get_status_markdown, outputs=hardware)

                with gr.Row():
                    preview = gr.Button("Preview", elem_classes=["preview-btn"])
                    generate = gr.Button("Generate video", variant="primary", elem_classes=["generate-btn"])

            with gr.Column(scale=5, elem_classes=["result-card"]):
                gr.HTML("<div class='result-title'>Your video</div><div class='result-subtitle'>Your preview and live render progress will appear here.</div>")
                output = gr.Video(label="Your video", autoplay=True, height=430)
                gr.HTML("<div class='section-heading compact'>Live generation status</div><div class='section-help'>The status updates as the GPU prepares, renders, and exports your video.</div>")
                status = gr.Textbox(label="Live generation status", value="Ready when you are. Describe a scene, choose your setup, then generate.", lines=11, interactive=False, elem_classes=["status-box"])
                with gr.Group(elem_classes=["plan-box"]):
                    plan_view = gr.Markdown("### Video plan\n\nChoose your settings and select **Preview** to review the render plan.")

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
                dialogue_audio,
                voice_mode,
                voice_language,
                reference_image,
                consistency,
                negative,
                seed,
            ],
            [output, plan_view, status],
        )

    return app


if __name__ == "__main__":
    create_app().launch(
        server_name="0.0.0.0",
        server_port=7860,
        inbrowser=False,
        show_error=True,
        theme=THEME,
        css=CSS,
    )
