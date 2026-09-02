"""RTX 4080 Moon Cookie studio powered by HunyuanVideo-1.5."""
from __future__ import annotations

import gradio as gr

from engine.hunyuan_generator import (
    ASPECT_RATIOS,
    DEFAULT_HUNYUAN_PRESET,
    HUNYUAN_PRESETS,
    HunyuanVideoGenerator,
    model_storage_summary,
)

GENERATOR = HunyuanVideoGenerator()

DEFAULT_NEGATIVE = (
    "character redesign, different character, changing costume, changing face, duplicate character, "
    "extra limbs, deformed hands, warped anatomy, flicker, jitter, morphing, blurry face, text, watermark, logo"
)

MOON_COOKIE_LOCK = (
    "Keep the exact same Moon Cookie character design from the reference image: same face, eyes, body proportions, "
    "colors, clothing/accessories, clay/cartoon material style and lighting language. Do not redesign the character. "
    "One clear action only. Preserve identity from the first frame to the last frame."
)

CSS = """
.gradio-container { max-width: 1180px !important; }
.hero { padding:24px 28px; border-radius:20px; margin-bottom:18px; border:1px solid rgba(120,120,150,.18); }
.hero h1 { margin:0 0 8px; font-size:34px; }
.note { padding:12px 14px; border-radius:12px; background:rgba(99,102,241,.08); line-height:1.45; }
"""


def generate(reference, action, consistency, negative, aspect, preset, seed, overlap, save_pre_sr, progress=gr.Progress()):
    if reference is None:
        return None, "Add the accepted Moon Cookie reference image first."
    if not action or not action.strip():
        return None, "Describe one clear 5-second action."

    lock = (consistency or MOON_COOKIE_LOCK).strip()
    prompt = f"{lock}\n\nSHOT ACTION: {action.strip()}\n\nCreate one continuous coherent shot with no cuts."
    logs: list[str] = []

    def callback(message: str, value: float):
        logs.append(message)
        progress(value, desc=message)

    try:
        path = GENERATOR.generate_image_to_video(
            prompt=prompt,
            image=reference,
            negative_prompt=negative or DEFAULT_NEGATIVE,
            aspect_ratio=aspect,
            preset=preset,
            seed=int(seed),
            overlap_group_offloading=bool(overlap),
            save_pre_sr_video=bool(save_pre_sr),
            progress_callback=callback,
        )
        return str(path), "\n".join(logs[-30:] + [f"Done: {path.name}"])
    except Exception as exc:
        return None, "\n".join(logs[-20:] + [f"Stopped: {type(exc).__name__}: {exc}"])


def create_app() -> gr.Blocks:
    with gr.Blocks(title="Moon Cookie • Hunyuan RTX 4080 Studio", css=CSS, analytics_enabled=False) as app:
        gr.HTML("""
        <section class='hero'>
          <h1>Moon Cookie • RTX 4080 Video Studio</h1>
          <p>HunyuanVideo-1.5 480p Image-to-Video Step-Distilled. Generate 5-second visual shots first, then add Tamil voices, music and lip-sync during final editing.</p>
        </section>
        """)
        gr.Markdown(model_storage_summary())
        gr.HTML("<div class='note'><b>Recommended episode workflow:</b> 30 final shots for ~2.5 minutes. Generate two attempts per shot (~60 raw generations). Use the accepted final frame from one scene as the next scene's reference.</div>")

        with gr.Row():
            with gr.Column(scale=5):
                reference = gr.Image(type="pil", label="Moon Cookie reference / previous accepted final frame")
                action = gr.Textbox(
                    label="What happens in this 5-second scene?",
                    lines=5,
                    placeholder="Example: Moon Cookie slowly opens the glowing cookie box, smiles with surprise, and looks toward the small window while Earth shines outside.",
                )
                consistency = gr.Textbox(label="Character lock", value=MOON_COOKIE_LOCK, lines=5)
                negative = gr.Textbox(label="Avoid", value=DEFAULT_NEGATIVE, lines=3)
                with gr.Row():
                    aspect = gr.Dropdown(list(ASPECT_RATIOS), value="16:9", label="Aspect ratio")
                    preset = gr.Dropdown(list(HUNYUAN_PRESETS), value=DEFAULT_HUNYUAN_PRESET, label="Render preset")
                with gr.Accordion("Advanced RTX 4080 settings", open=False):
                    seed = gr.Number(value=-1, precision=0, label="Seed (-1 = random)")
                    overlap = gr.Checkbox(value=False, label="Overlap group offloading (faster, uses more system RAM)")
                    save_pre_sr = gr.Checkbox(value=False, label="Also save native 480p video before SR")
                button = gr.Button("Generate Moon Cookie shot", variant="primary")

            with gr.Column(scale=5):
                output = gr.Video(label="Generated shot", autoplay=True)
                status = gr.Textbox(label="Generation log", lines=20, interactive=False)

        button.click(
            generate,
            [reference, action, consistency, negative, aspect, preset, seed, overlap, save_pre_sr],
            [output, status],
        )
    return app
