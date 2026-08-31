"""LTX Video Director Studio — premium local UI for RTX 4050 laptops."""
from __future__ import annotations

from datetime import datetime
from functools import partial
from pathlib import Path

import gradio as gr

from config import (
    ACTION_INTENSITY,
    ACTION_NEGATIVE_PROMPT,
    ACTION_STYLES,
    CAMERA_MOTIONS,
    CARTOON_STYLES,
    COMICS_NEGATIVE_PROMPT,
    COMICS_STYLES,
    DEFAULT_DURATION,
    DEFAULT_FPS,
    DEFAULT_GUIDANCE_SCALE,
    DEFAULT_NUM_INFERENCE_STEPS,
    DEFAULT_RESOLUTION,
    DEFAULT_STORY_SCENES,
    DURATION_PRESETS,
    EXPORT_PRESETS,
    LIGHTING_PRESETS,
    MAX_STORY_SCENES,
    NEGATIVE_PROMPT,
    OUTPUTS_DIR,
    REAL_WORLD_NEGATIVE_PROMPT,
    REAL_WORLD_STYLES,
    RESOLUTION_PRESETS,
    SHOT_PRESETS,
    STUDIO_EXAMPLES,
)
from engine.generator import VideoGenerator
from engine.memory_manager import get_status_markdown
from engine.prompt_builders import build_directed_prompt
from engine.storyboard import CartoonStoryGenerator, ContinuousSequenceGenerator, storyboard_markdown
from engine.video_processor import export_delivery

GENERATOR = VideoGenerator()
STORY_GENERATOR = CartoonStoryGenerator(GENERATOR)
SEQUENCE_GENERATOR = ContinuousSequenceGenerator(GENERATOR)
CSS_PATH = Path(__file__).parent / "static" / "style.css"
CUSTOM_CSS = CSS_PATH.read_text(encoding="utf-8") if CSS_PATH.exists() else ""


def _resolve(resolution: str, duration: str) -> tuple[int, int, int]:
    res = RESOLUTION_PRESETS[resolution]
    return res["width"], res["height"], DURATION_PRESETS[duration]


def _delivery(path: Path, preset: str, enhance_quality: bool = True) -> Path:
    target = EXPORT_PRESETS.get(preset)
    if target is None:
        return path
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = OUTPUTS_DIR / f"delivery_{target[0]}x{target[1]}_{stamp}.mp4"
    return export_delivery(path, out, *target, enhance_quality=enhance_quality)


def preview_image_sequence(target_duration, character_bible, style, prompt1, prompt2, prompt3, prompt4):
    steps = [p.strip() for p in [prompt1, prompt2, prompt3, prompt4] if p and p.strip()]
    if not steps:
        steps = ["Scene 1: Enter your starting prompt / action beat above"]
    lines = [
        f"### 🎬 Continuous Image Sequence Plan · {target_duration}",
        f"**Visual Style:** {style}",
        f"**Character Consistency Lock:** {character_bible or 'Not specified'}",
        "",
        "---",
    ]
    for i, p in enumerate(steps, start=1):
        lines.append(f"**Step {i} ({'Initial Keyframe' if i==1 else 'Keyframe or Auto-Chained Frame'}):** {p}")
    lines.append("")
    lines.append("> 💡 **Continuous Animation Tip:** Upload high-quality cartoon images for each scene, or leave subsequent images blank to let the AI automatically extract and carry over the previous clip's ending frame.")
    return "\n\n".join(lines)


def generate_image_sequence(
    target_duration: str,
    character_bible: str,
    style: str,
    quality_preset: str,
    resolution: str,
    negative: str,
    seed: int,
    steps: int,
    guidance: float,
    # Step 1
    img1, prompt1, camera1,
    # Step 2
    img2, prompt2, camera2,
    # Step 3
    img3, prompt3, camera3,
    # Step 4
    img4, prompt4, camera4,
    progress=gr.Progress(),
):
    raw_steps = [
        {"image": img1, "prompt": prompt1, "camera": camera1},
        {"image": img2, "prompt": prompt2, "camera": camera2},
        {"image": img3, "prompt": prompt3, "camera": camera3},
        {"image": img4, "prompt": prompt4, "camera": camera4},
    ]
    active_steps = [s for s in raw_steps if (s["prompt"] and s["prompt"].strip()) or s["image"] is not None]
    if not active_steps:
        raise gr.Error("Please enter a prompt or upload an image for at least Scene 1.")

    # Determine frames per step based on target continuous duration
    if "8.0" in target_duration:
        # 8s target: 2 scenes of 121 frames (~4.0s each) = 8.0s continuous video
        target_step_count = min(len(active_steps), 2)
        frames_per_step = 121
    elif "10.0" in target_duration:
        # 10s target: 3 scenes of 97 frames (~3.2s each) = ~9.7s continuous video
        target_step_count = min(len(active_steps), 3)
        frames_per_step = 97
    else:
        target_step_count = len(active_steps)
        frames_per_step = 121

    steps_to_run = active_steps[:target_step_count]
    for s in steps_to_run:
        s["frames"] = frames_per_step

    res = RESOLUTION_PRESETS[resolution]
    width, height = res["width"], res["height"]
    
    logs: list[str] = [f"Starting {len(steps_to_run)}-step continuous cartoon animation ({target_duration})..."]
    callback = _progress_bridge(progress, logs)

    final_path, individual_clips = SEQUENCE_GENERATOR.generate_sequence(
        sequence_steps=steps_to_run,
        character_bible=character_bible,
        style_name=style,
        width=width,
        height=height,
        num_inference_steps=int(steps),
        guidance_scale=float(guidance),
        negative_prompt=negative,
        seed=int(seed),
        progress_callback=callback,
    )

    delivered_path = _delivery(final_path, quality_preset, enhance_quality=True)
    total_sec = (frames_per_step / DEFAULT_FPS) * len(steps_to_run)
    logs.append(f"✅ Generated {len(steps_to_run)} continuous clips · Total runtime: {total_sec:.1f}s")
    logs.append(f"🎬 Saved high quality continuous video at: {delivered_path.name}")

    return str(delivered_path), "\n".join(logs[-20:])



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
    total = (frames / DEFAULT_FPS) * int(scene_count)
    logs.append(f"Approx story duration: {total:.1f}s before joins/encoding adjustments")
    return str(final), "\n".join(logs[-20:])


def _directed_prompt(
    mode,
    subject,
    action,
    environment,
    style,
    shot,
    camera,
    lighting,
    intensity,
    extra_details,
    dialogue_audio,
    reference_image,
    duration,
):
    frames = DURATION_PRESETS[duration]
    return build_directed_prompt(
        mode=mode,
        subject=subject,
        action=action,
        environment=environment,
        style_name=style,
        shot=shot,
        camera=camera,
        lighting=lighting,
        intensity_name=intensity,
        extra_details=extra_details,
        dialogue_audio=dialogue_audio,
        has_reference=reference_image is not None,
        duration_seconds=frames / DEFAULT_FPS,
    )


def preview_directed(
    mode,
    subject,
    action,
    environment,
    style,
    shot,
    camera,
    lighting,
    intensity,
    extra_details,
    dialogue_audio,
    reference_image,
    duration,
):
    if not (subject or action):
        return "Add a subject and/or action to build the directed prompt."
    return _directed_prompt(
        mode, subject, action, environment, style, shot, camera, lighting,
        intensity, extra_details, dialogue_audio, reference_image, duration,
    )


def generate_directed(
    mode,
    subject,
    action,
    environment,
    style,
    shot,
    camera,
    lighting,
    intensity,
    extra_details,
    dialogue_audio,
    reference_image,
    negative,
    resolution,
    duration,
    steps,
    guidance,
    seed,
    export_preset,
    progress=gr.Progress(),
):
    if not (subject or action):
        raise gr.Error("Add a subject and an action before generating.")
    prompt = _directed_prompt(
        mode, subject, action, environment, style, shot, camera, lighting,
        intensity, extra_details, dialogue_audio, reference_image, duration,
    )
    width, height, frames = _resolve(resolution, duration)
    logs: list[str] = [f"Director prompt built for {mode.replace('_', ' ')} mode."]
    callback = _progress_bridge(progress, logs)
    if reference_image is None:
        path = GENERATOR.generate_text_to_video(
            prompt, negative, width, height, frames, int(steps), float(guidance), int(seed), callback
        )
    else:
        path = GENERATOR.generate_image_to_video(
            prompt, reference_image, negative, width, height, frames, int(steps), float(guidance), int(seed), callback
        )
    final = _delivery(Path(path), export_preset)
    return str(final), prompt, "\n".join(logs[-14:])


def load_studio_example(mode: str, name: str):
    data = STUDIO_EXAMPLES.get(mode, {}).get(name, {})
    return (
        data.get("subject", ""),
        data.get("action", ""),
        data.get("environment", ""),
        data.get("extra", ""),
    )


def _build_directed_studio(
    mode: str,
    heading: str,
    description: str,
    styles: dict,
    negative_default: str,
    subject_label: str,
    action_label: str,
    environment_label: str,
    style_label: str,
    extra_label: str,
    action_mode: bool = False,
):
    gr.Markdown(f"## {heading}\n\n{description}")
    mode_state = gr.State(mode)
    with gr.Row(equal_height=False):
        with gr.Column(scale=5):
            examples = STUDIO_EXAMPLES.get(mode, {})
            example = gr.Dropdown(
                choices=[""] + list(examples),
                value="",
                label="Quick starting example",
                info="Optional — load a professionally structured starting scene, then edit it.",
            )
            reference_image = gr.Image(
                type="pil",
                label="Optional reference image — identity / composition anchor",
                height=220,
            )
            subject = gr.Textbox(label=subject_label, lines=2)
            action = gr.Textbox(
                label=action_label,
                lines=3,
                placeholder="Describe one clear physical action for short clips. Use observable verbs.",
            )
            environment = gr.Textbox(label=environment_label, lines=2)
            with gr.Row():
                style = gr.Dropdown(list(styles), value=next(iter(styles)), label=style_label)
                shot = gr.Dropdown(SHOT_PRESETS, value="Medium shot", label="Shot / framing")
            with gr.Row():
                camera = gr.Dropdown(CAMERA_MOTIONS, value="Gentle dolly in", label="Camera movement")
                lighting = gr.Dropdown(LIGHTING_PRESETS, value="Soft natural daylight", label="Lighting")
            intensity = gr.Dropdown(
                list(ACTION_INTENSITY),
                value="Dynamic",
                label="Action intensity",
                visible=action_mode,
            )
            extra_details = gr.Textbox(
                label=extra_label,
                lines=3,
                placeholder="Materials, costume lock, product detail, blocking, atmosphere, continuity notes…",
            )
            dialogue_audio = gr.Textbox(
                label="Optional dialogue / ambience / sound direction",
                lines=2,
                placeholder="Example: soft city ambience; no dialogue. Or: she says, “We made it.” in a calm voice.",
            )
            negative = gr.Textbox(label="Avoid / negative prompt", value=negative_default, lines=2)
            with gr.Row():
                resolution = gr.Dropdown(list(RESOLUTION_PRESETS), value=DEFAULT_RESOLUTION, label="Native generation size")
                duration = gr.Dropdown(list(DURATION_PRESETS), value=DEFAULT_DURATION, label="Clip duration")
            with gr.Row():
                export_preset = gr.Dropdown(list(EXPORT_PRESETS), value="Native MP4", label="Download size")
                seed = gr.Number(value=-1, precision=0, label="Seed (-1 random)")
            with gr.Accordion("Advanced generation", open=False):
                steps = gr.Slider(8, 40, value=DEFAULT_NUM_INFERENCE_STEPS, step=1, label="Inference steps")
                guidance = gr.Slider(1.0, 7.0, value=DEFAULT_GUIDANCE_SCALE, step=0.1, label="Guidance")
            with gr.Row():
                preview_btn = gr.Button("Build director prompt", elem_classes=["soft-btn"])
                generate_btn = gr.Button("Generate video", variant="primary", elem_classes=["primary-action"])

        with gr.Column(scale=6):
            prompt_preview = gr.Textbox(
                label="Director prompt sent to LTX",
                lines=10,
                interactive=False,
                elem_classes=["director-prompt"],
            )
            output = gr.Video(label=f"{heading} output", autoplay=True, height=390)
            status = gr.Textbox(label="Generation status", lines=8, interactive=False)

    example.change(
        fn=partial(load_studio_example, mode),
        inputs=example,
        outputs=[subject, action, environment, extra_details],
    )
    preview_inputs = [
        mode_state, subject, action, environment, style, shot, camera, lighting,
        intensity, extra_details, dialogue_audio, reference_image, duration,
    ]
    preview_btn.click(preview_directed, preview_inputs, prompt_preview)
    generate_btn.click(
        generate_directed,
        [
            mode_state, subject, action, environment, style, shot, camera, lighting,
            intensity, extra_details, dialogue_audio, reference_image, negative,
            resolution, duration, steps, guidance, seed, export_preset,
        ],
        [output, prompt_preview, status],
    )


def create_app() -> gr.Blocks:
    theme = gr.themes.Base(
        primary_hue=gr.themes.colors.violet,
        secondary_hue=gr.themes.colors.cyan,
        neutral_hue=gr.themes.colors.slate,
    )

    with gr.Blocks(title="LTX Video Director Studio", theme=theme, css=CUSTOM_CSS) as app:
        gr.HTML("""
        <section class="hero-shell">
          <div class="hero-kicker">LOCAL • OFFLINE AFTER FIRST DOWNLOAD • RTX 4050 PROFILE</div>
          <h1>LTX <span>Video Director Studio</span></h1>
          <p>Create continuous 8s–10s cartoon videos from prompt & image sequences, or direct scenes for Comics, Real-World, Action, and General creative workflows.</p>
          <div class="hero-pills"><b>🎬 Continuous Sequence (8s-10s)</b><b>✨ General</b><b>📚 Comics</b><b>🌍 Real World</b><b>⚡ Action</b><b>🧸 Cartoon Story</b><b>2B LTX • 8-bit</b></div>
        </section>
        """)

        with gr.Row():
            hardware = gr.Markdown(get_status_markdown(), elem_classes=["status-card"])
            refresh = gr.Button("Refresh hardware", elem_classes=["soft-btn"])
            refresh.click(get_status_markdown, outputs=hardware)

        gr.HTML("""
        <section class="mode-grid">
          <article><span>🎬</span><b>Continuous Sequence</b><small>8s-10s image & prompt chain</small></article>
          <article><span>✨</span><b>General</b><small>Free-form T2V / I2V</small></article>
          <article><span>📚</span><b>Comics</b><small>Ink, manga, graphic novel</small></article>
          <article><span>🌍</span><b>Real World</b><small>Live action & commercials</small></article>
          <article><span>⚡</span><b>Action</b><small>Dynamic choreography</small></article>
        </section>
        """)

        with gr.Tabs():
            with gr.Tab("🎬 Continuous Image Sequence Studio (8s - 10s)"):
                gr.Markdown(
                    "## 🎬 Continuous Cartoon Image Sequence Studio\n\n"
                    "Enter your cartoon scene prompts and upload corresponding keyframe cartoon images step by step. "
                    "The AI maintains seamless visual character continuity across scenes and joins them into a high-quality **8 to 10 second continuous video**."
                )
                with gr.Row(equal_height=False):
                    with gr.Column(scale=6):
                        with gr.Group():
                            with gr.Row():
                                seq_target_duration = gr.Dropdown(
                                    choices=[
                                        "8.0 Seconds Continuous Video (~241 frames • 2 scenes x 4.0s)",
                                        "10.0 Seconds Continuous Video (~300 frames • 3 scenes x 3.3s)",
                                        "Custom Sequence (4.0s per active scene)",
                                    ],
                                    value="8.0 Seconds Continuous Video (~241 frames • 2 scenes x 4.0s)",
                                    label="⏱️ Target Total Continuous Duration",
                                )
                                seq_style = gr.Dropdown(
                                    list(CARTOON_STYLES),
                                    value="Premium 3D Kids Animation",
                                    label="🎨 Cartoon Visual Style",
                                )
                            with gr.Row():
                                seq_quality = gr.Dropdown(
                                    ["1080p Landscape", "720p Landscape", "Native MP4", "1080p Portrait", "1080p Square"],
                                    value="1080p Landscape",
                                    label="✨ High-Quality Delivery Mode (Lanczos + Crisp Edge Enhancer)",
                                )
                                seq_resolution = gr.Dropdown(
                                    list(RESOLUTION_PRESETS),
                                    value=DEFAULT_RESOLUTION,
                                    label="Native RTX 4050 Render Size",
                                )
                            seq_character_bible = gr.Textbox(
                                label="🔒 Character Continuity Lock (Bible)",
                                placeholder="Describe main character appearance locked across all scenes: e.g. Milo the orange fox with green eyes, teal scarf, and brown boots.",
                                lines=2,
                            )

                        gr.Markdown("### 🎞️ Step-by-Step Sequence Beats & Keyframes")
                        with gr.Accordion("🎬 Scene 1 (Starting Keyframe / Action)", open=True):
                            with gr.Row():
                                img1 = gr.Image(type="pil", label="Keyframe Image 1 (Optional anchor)", height=150)
                                with gr.Column():
                                    prompt1 = gr.Textbox(
                                        label="Scene 1 Action Beat",
                                        placeholder="E.g. A cheerful cartoon puppy plays with a red ball in the sunny backyard.",
                                        lines=2,
                                    )
                                    camera1 = gr.Dropdown(CAMERA_MOTIONS, value="Gentle dolly in", label="Camera Motion 1")

                        with gr.Accordion("🎬 Scene 2 (Continuation 1)", open=True):
                            with gr.Row():
                                img2 = gr.Image(type="pil", label="Keyframe Image 2 (Optional / Auto-continues from Scene 1)", height=150)
                                with gr.Column():
                                    prompt2 = gr.Textbox(
                                        label="Scene 2 Action Beat",
                                        placeholder="E.g. The puppy chases the rolling ball through the garden flowers.",
                                        lines=2,
                                    )
                                    camera2 = gr.Dropdown(CAMERA_MOTIONS, value="Stable lateral tracking", label="Camera Motion 2")

                        with gr.Accordion("🎬 Scene 3 (Continuation 2)", open=False):
                            with gr.Row():
                                img3 = gr.Image(type="pil", label="Keyframe Image 3 (Optional / Auto-continues from Scene 2)", height=150)
                                with gr.Column():
                                    prompt3 = gr.Textbox(
                                        label="Scene 3 Action Beat",
                                        placeholder="E.g. The puppy catches the ball happily and wags its tail.",
                                        lines=2,
                                    )
                                    camera3 = gr.Dropdown(CAMERA_MOTIONS, value="Controlled orbit around the subject", label="Camera Motion 3")

                        with gr.Accordion("🎬 Scene 4 (Continuation 3)", open=False):
                            with gr.Row():
                                img4 = gr.Image(type="pil", label="Keyframe Image 4 (Optional / Auto-continues from Scene 3)", height=150)
                                with gr.Column():
                                    prompt4 = gr.Textbox(
                                        label="Scene 4 Action Beat",
                                        placeholder="E.g. The puppy runs back towards the porch with the ball.",
                                        lines=2,
                                    )
                                    camera4 = gr.Dropdown(CAMERA_MOTIONS, value="Slow cinematic pan", label="Camera Motion 4")

                        with gr.Accordion("Advanced Parameters", open=False):
                            seq_negative = gr.Textbox(label="Negative Prompt", value=NEGATIVE_PROMPT, lines=2)
                            with gr.Row():
                                seq_steps = gr.Slider(8, 40, value=DEFAULT_NUM_INFERENCE_STEPS, step=1, label="Inference steps")
                                seq_guidance = gr.Slider(1.0, 7.0, value=DEFAULT_GUIDANCE_SCALE, step=0.1, label="Guidance Scale")
                                seq_seed = gr.Number(value=-1, precision=0, label="Seed (-1 random)")

                        with gr.Row():
                            seq_preview_btn = gr.Button("📋 Preview Sequence Plan", elem_classes=["soft-btn"])
                            seq_generate_btn = gr.Button("🚀 Generate Continuous 8s-10s Video", variant="primary", elem_classes=["primary-action"])

                    with gr.Column(scale=5):
                        seq_preview_md = gr.Markdown("### Sequence plan and timeline will appear here…", elem_classes=["sequence-card"])
                        seq_output_video = gr.Video(label="🎬 Continuous Video Master Output", autoplay=True, height=420)
                        seq_status_log = gr.Textbox(label="Generation Progress & QC Status", lines=9, interactive=False)

                seq_preview_btn.click(
                    preview_image_sequence,
                    [seq_target_duration, seq_character_bible, seq_style, prompt1, prompt2, prompt3, prompt4],
                    seq_preview_md,
                )
                seq_generate_btn.click(
                    generate_image_sequence,
                    [
                        seq_target_duration, seq_character_bible, seq_style, seq_quality, seq_resolution,
                        seq_negative, seq_seed, seq_steps, seq_guidance,
                        img1, prompt1, camera1,
                        img2, prompt2, camera2,
                        img3, prompt3, camera3,
                        img4, prompt4, camera4,
                    ],
                    [seq_output_video, seq_status_log],
                )

            with gr.Tab("✨ General Studio"):
                gr.Markdown("## General Video Studio\n\nUse this when you already know exactly what you want to prompt. Uploading an image automatically switches generation to Image-to-Video.")
                with gr.Row(equal_height=False):
                    with gr.Column(scale=5):
                        image = gr.Image(type="pil", label="Optional reference image", height=220)
                        prompt = gr.Textbox(
                            label="Video description",
                            placeholder="Describe subject, action, environment, camera movement, lighting and visual style…",
                            lines=7,
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

            with gr.Tab("📚 Comics Studio"):
                _build_directed_studio(
                    mode="comics",
                    heading="Comics Video Studio",
                    description="For motion comics, manga, graphic novels and cel-shaded comic scenes. The director prompt locks line work, silhouette, costume and palette while keeping the shot readable.",
                    styles=COMICS_STYLES,
                    negative_default=COMICS_NEGATIVE_PROMPT,
                    subject_label="Character / comic subject",
                    action_label="Comic action beat",
                    environment_label="Comic world / location",
                    style_label="Comic style",
                    extra_label="Continuity / costume / art-direction notes",
                )

            with gr.Tab("🌍 Real-World Studio"):
                _build_directed_studio(
                    mode="real_world",
                    heading="Real-World Video Studio",
                    description="For cinematic live action, documentary, products, travel, presenters and lifestyle scenes. The prompt emphasizes realistic anatomy, materials, weight, inertia and physically plausible camera behavior.",
                    styles=REAL_WORLD_STYLES,
                    negative_default=REAL_WORLD_NEGATIVE_PROMPT,
                    subject_label="Person / product / real-world subject",
                    action_label="Physical action / performance",
                    environment_label="Location / time / real-world setting",
                    style_label="Realism profile",
                    extra_label="Material / blocking / realism details",
                )

            with gr.Tab("⚡ Action Studio"):
                _build_directed_studio(
                    mode="action",
                    heading="Action Video Studio",
                    description="For sports, chases, parkour, martial arts and adventure shots. Short clips are deliberately limited to one main action beat and one camera move so the motion stays understandable instead of becoming chaotic.",
                    styles=ACTION_STYLES,
                    negative_default=ACTION_NEGATIVE_PROMPT,
                    subject_label="Hero / athlete / vehicle / action subject",
                    action_label="Primary action beat",
                    environment_label="Action location / terrain",
                    style_label="Action style",
                    extra_label="Choreography / contact / trajectory notes",
                    action_mode=True,
                )

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

**Start here:** 384×224, 121 frames (~4.0 s), 20 steps, one scene at a time. If stable, try 512×288 or 161/193 frames. The 241-frame option remains the practical single-clip ceiling exposed by this UI.

**Comics:** use a strong reference image when a recurring hero must keep the same costume and face. Keep one main animated beat per short shot.

**Real World:** favor one physically plausible action, stable camera language, and natural lighting/material descriptions. Reference images work best when you describe the motion to add rather than re-describing the entire still.

**Action:** for ~4-second clips, use one stunt/action beat plus one camera move. Complex multi-event action is more reliable as multiple shots.

**Long cartoons:** use Cartoon Story Studio. It generates each short scene, extracts the final frame, conditions the next scene, and stitches the clips without loading the whole movie into RAM.

**1080p download:** the 1080p preset is a high-quality delivery resize. It does not invent native 1080p detail; native 1080p diffusion stays disabled in this RTX 4050 profile.

**Offline use:** `python run.py` installs missing Python packages and downloads the Hugging Face model the first time. After the model files are cached, generation can run without an API key.
                """)

        gr.HTML('<footer class="app-footer">Local creative modes • Prompt + image reference • RTX 4050 memory profile • No cloud generation required after model download</footer>')
    return app


if __name__ == "__main__":
    create_app().launch(server_name="127.0.0.1", server_port=7860, inbrowser=True, show_error=True)
