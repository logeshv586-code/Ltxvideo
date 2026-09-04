"""CogVideoX clip studio and persistent Moon Cookie Fox story workspace."""
from __future__ import annotations

from pathlib import Path

import gradio as gr

from config import STATIC_DIR
from easy_app import CSS, THEME
from engine.audio_processor import (
    add_audio_to_video,
    auto_narration_from_prompt,
    detect_voice_language,
    generate_speech,
    resolve_narration,
)
from engine.cogvideox_generator import COGVIDEOX_FPS, COGVIDEOX_FRAMES, GENERATOR as COGVIDEOX
from engine.longform import LongFormVideoGenerator, plan_story
from engine.optimized_generator import OptimizedVideoGenerator
from engine.video_processor import export_delivery
from engine.wan_generator import GENERATOR as WAN

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
STORY_DURATIONS = {
    "15 seconds": 15,
    "30 seconds": 30,
    "1 minute": 60,
    "5 minutes": 300,
    "10 minutes": 600,
}
VIDEO_MODELS = (
    ("LTX-Video — continuous stories / reference-free (T4-safe)", "ltx"),
    ("CogVideoX-5B — cinematic clips (slow T4-safe mode)", "cogvideox"),
    ("Wan2.1-T2V-1.3B — kids' cartoons / 480p", "wan"),
)


def update_creation_mode(creation_mode: str):
    """Avoid implying the fixed-duration clip model obeys story length."""
    if creation_mode == "Story assembly":
        return (
            gr.update(visible=True),
            "**Story assembly enabled:** scenes are rendered separately, then joined and trimmed to the selected final length. Long videos can take many hours.",
        )
    return (
        gr.update(visible=False),
        "**Single quality clip:** duration is fixed by the selected model (about 4 seconds for LTX, 5 seconds for Wan, and 6 seconds for CogVideoX). Select **Story assembly** to use 15 seconds through 10 minutes.",
    )


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


def generate_cog_story(
    prompt: str,
    duration: str,
    seed: int | float,
    custom_voice: str,
    voice_mode: str,
    language: str,
    progress=gr.Progress(),
):
    """Render each chronological event, join the scenes, then narrate it."""
    if not prompt or not prompt.strip():
        return None, "Describe the chronological story first."
    if detect_voice_language(prompt) == "Tamil":
        return None, "CogVideoX-5B requires an English visual prompt. Tamil narration is still supported in Voice & narration."
    seconds = STORY_DURATIONS.get(duration, 15)
    logs: list[str] = []

    def callback(message: str, value: float) -> None:
        logs.append(message)
        progress(value, desc=message)

    try:
        path, beats = COGVIDEOX.generate_story(prompt, seconds, seed, callback)
        # Automatic narration covers the assembled story, not only its opening
        # event. Custom narration remains authoritative when selected.
        narration_prompt = prompt
        if voice_mode == "Automatic from prompt":
            narration = " ".join(auto_narration_from_prompt(beat, max_words=16) for beat in beats)
            narration_prompt = f"Narration: {narration}"
        path, voice_result = _attach_narration(path, narration_prompt, custom_voice, voice_mode, language)
        scene_log = [f"Scene {index + 1}: {beat}" for index, beat in enumerate(beats)]
        return str(path), "\n".join(scene_log + logs[-24:] + [voice_result, f"Done: {path.name}"])
    except Exception as exc:
        return None, "\n".join(logs[-24:] + [f"Stopped: {type(exc).__name__}: {exc}"])


def generate_cog_request(
    creation_mode: str,
    prompt: str,
    duration: str,
    seed: int | float,
    custom_voice: str,
    voice_mode: str,
    language: str,
    progress=gr.Progress(),
):
    if creation_mode == "Story assembly":
        return generate_cog_story(prompt, duration, seed, custom_voice, voice_mode, language, progress)
    return generate_cog(prompt, seed, custom_voice, voice_mode, language, progress)


def generate_wan(prompt: str, seed: int | float, custom_voice: str, voice_mode: str, language: str, progress=gr.Progress()):
    if not prompt or not prompt.strip():
        return None, "Describe the Wan2.1 clip first."
    logs: list[str] = []

    def callback(message: str, value: float) -> None:
        logs.append(message)
        progress(value, desc=message)

    try:
        path = WAN.generate(prompt, seed, callback)
        path, voice_result = _attach_narration(path, prompt, custom_voice, voice_mode, language)
        return str(path), "\n".join(logs[-20:] + [voice_result, f"Done: {path.name}"])
    except Exception as exc:
        return None, "\n".join(logs[-20:] + [f"Stopped: {type(exc).__name__}: {exc}"])


def generate_wan_story(
    prompt: str,
    duration: str,
    seed: int | float,
    custom_voice: str,
    voice_mode: str,
    language: str,
    progress=gr.Progress(),
):
    if not prompt or not prompt.strip():
        return None, "Describe the chronological story first."
    seconds = STORY_DURATIONS.get(duration, 15)
    logs: list[str] = []

    def callback(message: str, value: float) -> None:
        logs.append(message)
        progress(value, desc=message)

    try:
        path, beats = WAN.generate_story(prompt, seconds, seed, callback)
        narration_prompt = prompt
        if voice_mode == "Automatic from prompt":
            narration = " ".join(auto_narration_from_prompt(beat, max_words=16) for beat in beats)
            narration_prompt = f"Narration: {narration}"
        path, voice_result = _attach_narration(path, narration_prompt, custom_voice, voice_mode, language)
        scene_log = [f"Scene {index + 1}: {beat}" for index, beat in enumerate(beats)]
        return str(path), "\n".join(scene_log + logs[-24:] + [voice_result, f"Done: {path.name}"])
    except Exception as exc:
        return None, "\n".join(logs[-24:] + [f"Stopped: {type(exc).__name__}: {exc}"])


def generate_ltx_request(
    creation_mode: str,
    prompt: str,
    duration: str,
    seed: int | float,
    custom_voice: str,
    voice_mode: str,
    language: str,
    progress=gr.Progress(),
):
    """Use LTX when the user prefers connected, T4-safe story generation."""
    if not prompt or not prompt.strip():
        return None, "Describe the LTX video first."
    mode = "Continuous Video" if creation_mode == "Story assembly" else "Single Clip"
    plan = plan_story(
        prompt,
        duration,
        "High",
        "Landscape • 1280×720 • 16:9",
        generation_mode=mode,
        clip_length_label="4 seconds • Recommended",
    )
    logs: list[str] = []

    def callback(message: str, value: float) -> None:
        logs.append(message)
        progress(value, desc=message)

    try:
        raw = FOX_LONGFORM.generate(
            plan=plan,
            style_prompt="cinematic, coherent motion, clean detail",
            character_lock="",
            reference_image=None,
            negative_prompt="flicker, morphing, duplicate subjects, extra limbs, blurry, text, watermark",
            seed=int(seed),
            progress_callback=callback,
        )
        delivery = raw.with_name(raw.stem + "_ltx_1080p.mp4")
        export_delivery(
            raw,
            delivery,
            1920,
            1080,
            enhance_quality=True,
            target_fps=plan.profile.fps,
            duration_seconds=plan.target_seconds,
        )
        delivery, voice_result = _attach_narration(delivery, prompt, custom_voice, voice_mode, language)
        return str(delivery), "\n".join(logs[-24:] + [voice_result, f"Done: {delivery.name}"])
    except Exception as exc:
        return None, "\n".join(logs[-24:] + [f"Stopped: {type(exc).__name__}: {exc}"])


def generate_video_request(
    model: str,
    creation_mode: str,
    prompt: str,
    duration: str,
    seed: int | float,
    custom_voice: str,
    voice_mode: str,
    language: str,
    progress=gr.Progress(),
):
    if model == "ltx":
        return generate_ltx_request(creation_mode, prompt, duration, seed, custom_voice, voice_mode, language, progress)
    if model == "wan":
        if creation_mode == "Story assembly":
            return generate_wan_story(prompt, duration, seed, custom_voice, voice_mode, language, progress)
        return generate_wan(prompt, seed, custom_voice, voice_mode, language, progress)
    return generate_cog_request(creation_mode, prompt, duration, seed, custom_voice, voice_mode, language, progress)


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
    with gr.Blocks(
        title="LTX Creator — CogVideoX & Moon Cookie Fox",
        analytics_enabled=False,
        theme=THEME,
        css=CSS,
    ) as app:
        gr.HTML("""
          <header class="studio-header"><div class="brand-mark">LTX</div><div class="brand-name">Quality Clip Studio</div>
          <div class="steps"><span class="step active"><b>1</b> Choose engine</span><i class="step-line"></i><span class="step"><b>2</b> Create</span></div></header>
        """)
        with gr.Tabs():
            with gr.Tab("Video models"):
                cog_ready, cog_status = COGVIDEOX.readiness()
                wan_ready, wan_status = WAN.readiness()
                gr.Markdown("**Choose the engine for the job.** LTX-Video is the T4-safe choice for connected multi-part stories. CogVideoX is slower and optimized for higher-detail independent cinematic clips. Wan2.1 1.3B is the practical 480p cartoon option before 1080p delivery export.")
                gr.Markdown(f"**CogVideoX status:** {'Ready' if cog_ready else 'Setup required'} — {cog_status}\n\n**Wan2.1 status:** {'Ready' if wan_ready else 'Setup required'} — {wan_status}\n\n**Story assembly:** write events in order using *then*, *after that*, or *finally*. Each scene is rendered separately, combined in that sequence, trimmed to the selected length, and given one final voice track.")
                with gr.Row():
                    with gr.Column(scale=6, elem_classes=["creator-card"]):
                        model = gr.Radio(VIDEO_MODELS, value="cogvideox", label="Video model")
                        creation_mode = gr.Radio(("Single quality clip", "Story assembly"), value="Single quality clip", label="Creation mode")
                        prompt = gr.Textbox(label="Visual prompt / chronological story", lines=8, placeholder="A fox child enters a moonlit garden. Then she finds a glowing cookie box. Finally she opens it and smiles toward the camera. Cinematic medium shot, gentle dolly in.")
                        story_duration = gr.Dropdown(
                            tuple(STORY_DURATIONS),
                            value="15 seconds",
                            label="Final length (Story assembly)",
                            visible=False,
                        )
                        duration_note = gr.Markdown(
                            "**Single quality clip:** duration is fixed by the selected model (about 4 seconds for LTX, 5 seconds for Wan, and 6 seconds for CogVideoX). Select **Story assembly** to use 15 seconds through 10 minutes."
                        )
                        seed = gr.Number(value=-1, precision=0, label="Seed (-1 = random)")
                        with gr.Accordion("Voice & narration", open=False):
                            custom_voice, voice_mode, language = _voice_controls()
                        generate = gr.Button("Generate selected model", variant="primary", elem_classes=["generate-btn"])
                    with gr.Column(scale=5, elem_classes=["result-card"]):
                        output = gr.Video(label="Final 1080p delivery", autoplay=True, height=420)
                        log = gr.Textbox(label="Generation status", lines=14, interactive=False, elem_classes=["status-box"])
                generate.click(
                    generate_video_request,
                    [model, creation_mode, prompt, story_duration, seed, custom_voice, voice_mode, language],
                    [output, log],
                )
                creation_mode.change(update_creation_mode, creation_mode, [story_duration, duration_note])

            with gr.Tab("Moon Cookie Fox continuity"):
                gr.Markdown("### Persistent Fox story mode\n\nThis tab applies the same Fox character and garden world to every prompt. For a continuous video, LTX carries tail frames from each accepted shot into the next one. Upload an approved reference frame to lock the opening scene even more strongly.")
                with gr.Row():
                    with gr.Column(scale=6, elem_classes=["creator-card"]):
                        action = gr.Textbox(label="What happens next?", lines=4, placeholder="The Fox carefully opens a glowing cookie box, sees a tiny moon inside, and smiles toward the clock tower.")
                        duration = gr.Dropdown(tuple(STORY_DURATIONS), value="15 seconds", label="Continuous story duration")
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
