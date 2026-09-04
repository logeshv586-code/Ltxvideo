import asyncio
import re
import subprocess
from pathlib import Path

try:
    import edge_tts
except ImportError:
    edge_tts = None

import imageio_ffmpeg

VOICE_BY_LANGUAGE = {
    "Tamil": "ta-IN-PallaviNeural",
    "English": "en-US-AriaNeural",
}
TAMIL_RANGE = re.compile(r"[\u0B80-\u0BFF]")
NARRATION_DIRECTIVE = re.compile(
    r"^\s*(?:narration|voiceover|voice|dialogue)\s*:\s*(.+?)\s*$",
    flags=re.IGNORECASE,
)
PROMPT_DIRECTIVE = re.compile(
    r"^\s*(?:style|look|camera|duration|aspect(?:\s+ratio)?|resolution|size|quality|fps|frame\s*rate)\s*:",
    flags=re.IGNORECASE,
)


def detect_voice_language(text: str, requested_language: str = "Auto detect") -> str:
    """Resolve the speech language without requiring a separate language picker."""
    if requested_language in VOICE_BY_LANGUAGE:
        return requested_language
    return "Tamil" if TAMIL_RANGE.search(text or "") else "English"


def auto_narration_from_prompt(prompt: str, max_words: int = 16) -> str:
    """Turn a visual prompt into a short, speakable line that fits a short clip.

    An explicit ``Narration:``, ``Voiceover:``, ``Voice:`` or ``Dialogue:`` line
    is always used verbatim. Otherwise the first story sentence is used after
    removing generation-only directives such as ``Style:`` and ``Camera:``.
    """
    lines = [line.strip() for line in (prompt or "").splitlines() if line.strip()]
    for line in lines:
        match = NARRATION_DIRECTIVE.match(line)
        if match:
            return match.group(1).strip()

    story = " ".join(line for line in lines if not PROMPT_DIRECTIVE.match(line))
    story = re.sub(r"\s+", " ", story).strip()
    if not story:
        return ""
    first_sentence = re.split(r"(?<=[.!?])\s+", story, maxsplit=1)[0]
    words = first_sentence.split()
    return " ".join(words[:max_words]).rstrip(" ,;:-")


def resolve_narration(
    prompt: str,
    custom_text: str = "",
    mode: str = "Automatic from prompt",
    requested_language: str = "Auto detect",
) -> tuple[str, str]:
    """Return short narration text and its language for the requested voice mode."""
    if mode == "No voice":
        return "", detect_voice_language(prompt, requested_language)
    text = (custom_text or "").strip() if mode == "Custom narration" else auto_narration_from_prompt(prompt)
    return text, detect_voice_language(text or prompt, requested_language)


def generate_speech(
    text: str,
    output_path: str | Path,
    language: str = "Auto detect",
    voice: str | None = None,
) -> str:
    """Generate Tamil or English narration with the matching Edge neural voice."""
    if edge_tts is None:
        raise RuntimeError("edge-tts is not installed. Run `pip install edge-tts`.")
    if not text or not text.strip():
        raise ValueError("Narration text is empty.")
    resolved_language = detect_voice_language(text, language)
    selected_voice = voice or VOICE_BY_LANGUAGE[resolved_language]

    async def _generate() -> None:
        communicate = edge_tts.Communicate(text.strip(), selected_voice)
        await communicate.save(str(output_path))

    asyncio.run(_generate())
    return resolved_language


def generate_tamil_tts(text: str, output_path: str | Path, voice: str = "ta-IN-PallaviNeural") -> None:
    """Backward-compatible Tamil-only wrapper for earlier callers."""
    generate_speech(text, output_path, language="Tamil", voice=voice)

def add_audio_to_video(video_path: str | Path, audio_path: str | Path, output_path: str | Path) -> None:
    """Mux audio and video together using ffmpeg."""
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [
        str(ffmpeg),
        "-y",
        "-i", str(video_path),
        "-i", str(audio_path),
        "-c:v", "copy",
        "-c:a", "aac",
        "-b:a", "192k",
        "-shortest",  # Trim audio to video length if it's longer
        str(output_path)
    ]
    
    # Run ffmpeg, suppress output unless it errors
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"FFmpeg muxing failed: {exc.stderr.decode()}") from exc
