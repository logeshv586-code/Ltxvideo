"""Mode-aware prompt directors for LTX video generation.

The builders turn easy form fields into one chronological shot description.
They deliberately keep short clips simple: one primary action and one camera
move, with reference-image mode focused on changes rather than restating the
input frame.
"""
from __future__ import annotations

from config import ACTION_INTENSITY, ACTION_STYLES, COMICS_STYLES, REAL_WORLD_STYLES

MAX_PROMPT_WORDS = 190


def _clean(value: str | None) -> str:
    return " ".join((value or "").strip().split())


def _sentence(value: str | None) -> str:
    text = _clean(value)
    if not text:
        return ""
    return text if text.endswith((".", "!", "?")) else f"{text}."


def _trim_words(text: str, max_words: int = MAX_PROMPT_WORDS) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text.strip()
    return " ".join(words[:max_words]).rstrip(",;:") + "."


def _reference_anchor(has_reference: bool) -> str:
    if not has_reference:
        return ""
    return (
        "Use the supplied reference image as the fixed visual identity and opening composition. "
        "Preserve the subject design, clothing, colors, proportions, product shape and major background layout; describe only the new motion and camera behavior. "
    )


def build_comics_prompt(
    subject: str,
    action: str,
    environment: str,
    style_name: str,
    shot: str,
    camera: str,
    lighting: str,
    extra_details: str = "",
    dialogue_audio: str = "",
    has_reference: bool = False,
    duration_seconds: float = 4.0,
) -> str:
    style = COMICS_STYLES.get(style_name, style_name)
    motion_scope = (
        "Keep this short shot to one clearly readable action beat and one controlled camera move."
        if duration_seconds <= 6.0
        else "Let the action progress in two clear chronological beats without a hard cut."
    )
    prompt = " ".join(
        part for part in [
            _reference_anchor(has_reference),
            _sentence(f"{subject} {action}" if subject else action),
            _sentence(f"The scene takes place in {environment}" if environment else ""),
            _sentence(f"Visual direction: {style}"),
            _sentence(f"Composition: {shot}"),
            _sentence(f"Camera: {camera}"),
            _sentence(f"Lighting and color: {lighting}"),
            _sentence(extra_details),
            _sentence(dialogue_audio),
            _sentence(motion_scope),
            "Maintain consistent line work, silhouette, costume, facial design and color palette throughout the shot.",
        ]
        if part
    )
    return _trim_words(prompt)


def build_real_world_prompt(
    subject: str,
    action: str,
    environment: str,
    style_name: str,
    shot: str,
    camera: str,
    lighting: str,
    extra_details: str = "",
    dialogue_audio: str = "",
    has_reference: bool = False,
    duration_seconds: float = 4.0,
) -> str:
    style = REAL_WORLD_STYLES.get(style_name, style_name)
    motion_scope = (
        "Use one primary physical action with a clear beginning and end; keep the camera move simple and physically plausible."
        if duration_seconds <= 6.0
        else "Use at most two sequential physical action beats with continuous spatial logic and no hard cut."
    )
    prompt = " ".join(
        part for part in [
            _reference_anchor(has_reference),
            _sentence(f"{subject} {action}" if subject else action),
            _sentence(f"Location and time: {environment}" if environment else ""),
            _sentence(f"Realism profile: {style}"),
            _sentence(f"Framing: {shot}"),
            _sentence(f"Camera behavior: {camera}"),
            _sentence(f"Lighting: {lighting}"),
            _sentence(extra_details),
            _sentence(dialogue_audio),
            _sentence(motion_scope),
            "Preserve natural anatomy, believable weight and inertia, realistic materials, consistent identity, and coherent background geometry.",
        ]
        if part
    )
    return _trim_words(prompt)


def build_action_prompt(
    subject: str,
    action: str,
    environment: str,
    style_name: str,
    shot: str,
    camera: str,
    lighting: str,
    intensity_name: str = "Dynamic",
    extra_details: str = "",
    dialogue_audio: str = "",
    has_reference: bool = False,
    duration_seconds: float = 4.0,
) -> str:
    style = ACTION_STYLES.get(style_name, style_name)
    intensity = ACTION_INTENSITY.get(intensity_name, intensity_name)
    if duration_seconds <= 6.0:
        timing = (
            "Action timing: establish the pose immediately, execute one primary action beat, then settle into a readable end state. "
            "Do not add a second unrelated stunt or camera move."
        )
    else:
        timing = (
            "Action timing: establish the geography, execute the primary action, then add one natural follow-through beat and finish in a readable end state."
        )
    prompt = " ".join(
        part for part in [
            _reference_anchor(has_reference),
            _sentence(f"{subject} {action}" if subject else action),
            _sentence(f"Action environment: {environment}" if environment else ""),
            _sentence(f"Direction: {style}"),
            _sentence(f"Energy: {intensity}"),
            _sentence(f"Framing: {shot}"),
            _sentence(f"Camera: {camera}"),
            _sentence(f"Lighting: {lighting}"),
            _sentence(extra_details),
            _sentence(dialogue_audio),
            _sentence(timing),
            "Keep body mechanics, contact points, momentum, object trajectories and screen direction physically coherent; prioritize readable choreography over chaos.",
        ]
        if part
    )
    return _trim_words(prompt)


def build_directed_prompt(
    mode: str,
    subject: str,
    action: str,
    environment: str,
    style_name: str,
    shot: str,
    camera: str,
    lighting: str,
    extra_details: str = "",
    dialogue_audio: str = "",
    has_reference: bool = False,
    duration_seconds: float = 4.0,
    intensity_name: str = "Dynamic",
) -> str:
    mode_key = _clean(mode).lower()
    common = dict(
        subject=subject,
        action=action,
        environment=environment,
        style_name=style_name,
        shot=shot,
        camera=camera,
        lighting=lighting,
        extra_details=extra_details,
        dialogue_audio=dialogue_audio,
        has_reference=has_reference,
        duration_seconds=duration_seconds,
    )
    if mode_key == "comics":
        return build_comics_prompt(**common)
    if mode_key == "real_world":
        return build_real_world_prompt(**common)
    if mode_key == "action":
        return build_action_prompt(**common, intensity_name=intensity_name)
    raise ValueError(f"Unknown studio mode: {mode}")
