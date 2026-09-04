import asyncio
from pathlib import Path
from engine.audio_processor import generate_tamil_tts

print("Generating audio...")
generate_tamil_tts("வணக்கம்! இது ஒரு பரிசோதனை.", "test_audio.mp3")
print("Audio generated successfully!" if Path("test_audio.mp3").exists() else "Audio generation failed.")
