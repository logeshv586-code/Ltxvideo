"""CogVideoX clip studio and persistent Moon Cookie Fox story workspace."""
from __future__ import annotations

from pathlib import Path

import gradio as gr

from config import STATIC_DIR
from easy_app import CSS, THEME
from engine.audio_processor import add_audio_to_video, detect_voice_language, generate_speech, resolve_narration
from engine.cogvideox_generator import COGVIDEOX_FPS, COGVIDEOX_FRAMES, GENERATOR as COGVIDEOX
from engine.longform import LongFormVideoGenerator, plan_story
from engine.optimized_generator import OptimizedVideoGenerator
from engine.video_processor import export_delivery

FOX_CHARACTER = (
    "Moon Cookie Fox is a small, friendly orange fox child with large warm brown eyes, a cream muzzle and chest, "
    "soft rounded 3D animated proportions, a tiny violet moon-shaped hair clip, and a magenta knitted skirt. "
    "Keep the exact same face, fur markings, proportions, clothing and color palette in every shot."
)
FOX_ENVIRONMENT = (
    "A dreamy Moon Cookie garden village at golden hour: rounded trimmed trees, pastel cottages, a tall whimsical "
    "moon-clock tower, blue hydrangeas and warm peach clouds. Preserve the same storybook 3D environment, lighting "
    "and left-to-right screen geography in every continuation."
)
FOX_STYLE = (
    "premium family-feature 3D animation, clean soft fur detail, polished materials, gentle cinematic global illumination, "
    "warm controlled color, stable facial acting and clear readable motion"
)
FOX_REFERENCE = STATIC_DIR / "moon-cookie-fox-reference-v1.png"
FOX_GENERATOR = OptimizedVideoGenerator()
FOX_LONGFORM = LongFormVideoGenerator(FOX_GENERATOR)


def _attach_narration(video: Path, prompt: str, custom_text: str, voice_mode: str, language: str) -> tuple[Path, str]:
    narration, resolved_language = resolve_narration(prompt, custom_text, voice_mode, language)
    if not narration:
        return video, "No narration selected."
    audio_path = video.with_suffix(".mp3")
    final = video.with_name(video.stem + "_audio.mp4")
    generate_speech(narration, audio_path, language=resolved_language)
    add_audio_to_video(video, audio_path, final)
    return final, f"{resolved_language} narration attached."


def generate_cog(prompt: str, seed: int | float, custom_voice: str, voice_mode: str, language: str, progress=gr.Progress()):
    if not prompt or not prompt.strip():
        return None, "Describe the CogVideoX clip first."
    if detect_voice_language(prompt) == "Tamil":
        return None, "CogVideoX-5B requires an English visual prompt. Tamil narration is still supported in Voice & narration."
    logs: list[str] = []

    def callback(message: str, value: float) -> None:
        logs.append(message)
        progress(value, desc=message)

    try:
        path = COGVIDEOX.generate(prompt, seed, callback)
        path, voice_result = _attach_narration(path, prompt, custom_voice, voice_mode, language)
        return str(path), "\n".join(logs[-20:] + [voice_result, f"Done: {path.name}"])
    except Exception as exc:
        return None, "\n".join(logs[-20:] + [f"Stopped: {type(exc).__name__}: {exc}"])


def generate_fox_story(
    action: str,
    duration: str,
    reference,
    character: str,
    environment: str,
    custom_voice: str,
    voice_mode: str,
    language: str,
    seed: int | float,
    progress=gr.Progress(),
):
    if not action or not action.strip():
        return None, "Describe the next Fox action first."
    character_lock = (character or FOX_CHARACTER).strip()
    world_lock = (environment or FOX_ENVIRONMENT).strip()
    story = f"{character_lock}\n\nEnvironment lock: {world_lock}\n\nNext story action: {action.strip()}"
    plan = plan_story(
        story,
        duration,
        "High",
        "Landscape â€¢ 1280×720 â€¢ 16:9",
        generation_mode="Continuous Video",
        clip_length_label="4 seconds • Recommended",
    )
    logs: list[str] = []

    def callback(message: str, value: float) -> None:
        logs.append(message)
        progress(value, desc=message)

    try:
        raw = FOX_LONGFORM.generate(
            plan=plan,
            style_prompt=FOX_STYLE,
            character_lock=f"{character_lock} Environment lock: {world_lock}",
            reference_image=reference,
            negative_prompt="character redesign, different costume, different environment, duplicate character, extra limbs, warped anatomy, flicker, morphing, blurry, text, watermark, logo",
            seed=int(seed),
            progress_callback=callback,
        )
        delivery = raw.with_name(raw.stem + "_fox_1080p.mp4")
        export_delivery(raw, delivery, 1920, 1080, enhance_quality=True, target_fps=plan.profile.fps, duration_seconds=plan.target_seconds)
        delivery, voice_result = _attach_narration(delivery, action, custom_voice, voice_mode, language)
        return str(delivery), "\n".join(logs[-24:] + [voice_result, f"Done: {delivery.name}"])
    except Exception as exc:
        return None, "\n".join(logs[-24:] + [f"Stopped: {type(exc).__name__}: {exc}"])


def _voice_controls() -> tuple[gr.Textbox, gr.Radio, gr.Dropdown]:
    custom_voice = gr.Textbox(label="Custom narration (optional)", lines=2, placeholder="Narration: Welcome to a new Moon Cookie story.")
    voice_mode = gr.Radio(("Automatic from prompt", "Custom narration", "No voice"), value="Automatic from prompt", label="Narration")
    language = gr.Dropdown(("Auto detect", "Tamil", "English"), value="Auto detect", label="Language")
    return custom_voice, voice_mode, language


def create_app() -> gr.Blocks:
    with gr.Blocks(title="LTX Creator — CogVideoX & Moon Cookie Fox", analytics_enabled=False) as app:
        gr.HTML("""
          <header class="studio-header"><div class="brand-mark">LTX</div><div class="brand-name">Quality Clip Studio</div>
          <div class="steps"><span class="step active"><b>1</b> Choose engine</span><i class="step-line"></i><span class="step"><b>2</b> Create</span></div></header>
        """)
        with gr.Tabs():
            with gr.Tab("CogVideoX-5B clips"):
                ready, status = COGVIDEOX.readiness()
                gr.Markdown(f"### CogVideoX-5B quality clip mode\n\n**Status:** {'Ready' if ready else 'Setup required'} — {status}\n\nWrite the **visual prompt in English**. Tamil and English narration remain available separately. This model renders **{COGVIDEOX_FRAMES} native frames at {COGVIDEOX_FPS} FPS** (about six seconds), then produces a 1080p YouTube delivery file. Each T4 receives an independent INT8-weight worker.")
                with gr.Row():
                    with gr.Column(scale=6, elem_classes=["creator-card"]):
                        prompt = gr.Textbox(label="Describe one cinematic clip", lines=8, placeholder="A fox child walks through a moonlit garden, pauses beside a glowing cookie box, then smiles toward the camera. Cinematic medium shot, gentle dolly in.")
                        seed = gr.Number(value=-1, precision=0, label="Seed (-1 = random)")
                        with gr.Accordion("Voice & narration", open=False):
                            custom_voice, voice_mode, language = _voice_controls()
                        generate = gr.Button("Generate CogVideoX 1080p clip", variant="primary", elem_classes=["generate-btn"])
                    with gr.Column(scale=5, elem_classes=["result-card"]):
                        output = gr.Video(label="CogVideoX delivery", autoplay=True, height=420)
                        log = gr.Textbox(label="Generation status", lines=14, interactive=False, elem_classes=["status-box"])
                generate.click(generate_cog, [prompt, seed, custom_voice, voice_mode, language], [output, log])

            with gr.Tab("Moon Cookie Fox continuity"):
                gr.Markdown("### Persistent Fox story mode\n\nThis tab applies the same Fox character and garden world to every prompt. For a continuous video, LTX carries tail frames from each accepted shot into the next one. Upload an approved reference frame to lock the opening scene even more strongly.")
                with gr.Row():
                    with gr.Column(scale=6, elem_classes=["creator-card"]):
                        action = gr.Textbox(label="What happens next?", lines=4, placeholder="The Fox carefully opens a glowing cookie box, sees a tiny moon inside, and smiles toward the clock tower.")
                        duration = gr.Dropdown(("15 seconds", "30 seconds", "1 minute"), value="15 seconds", label="Continuous story duration")
                        reference = gr.Image(
                            value=str(FOX_REFERENCE) if FOX_REFERENCE.exists() else None,
                            type="pil",
                            label="Moon Cookie Fox reference (replace with an approved final frame if needed)",
                        )
                        with gr.Accordion("Moon Cookie Fox bible", open=False):
                            character = gr.Textbox(label="Character lock", value=FOX_CHARACTER, lines=5)
                            environment = gr.Textbox(label="Environment lock", value=FOX_ENVIRONMENT, lines=5)
                        with gr.Accordion("Voice & narration", open=False):
                            fox_voice, fox_mode, fox_language = _voice_controls()
                        fox_seed = gr.Number(value=-1, precision=0, label="Seed (-1 = random)")
                        fox_generate = gr.Button("Generate continuing Fox story", variant="primary", elem_classes=["generate-btn"])
                    with gr.Column(scale=5, elem_classes=["result-card"]):
                        fox_output = gr.Video(label="Moon Cookie Fox 1080p delivery", autoplay=True, height=420)
                        fox_log = gr.Textbox(label="Story generation status", lines=14, interactive=False, elem_classes=["status-box"])
                fox_generate.click(
                    generate_fox_story,
                    [action, duration, reference, character, environment, fox_voice, fox_mode, fox_language, fox_seed],
                    [fox_output, fox_log],
                )
    return app
