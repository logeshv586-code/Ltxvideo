import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.resolve()))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from engine.generator import VideoGenerator
from engine.hardware_profiles import get_active_hardware_profile
import config

print('=== Starting Ltxvideo Test Generation ===')
profile = get_active_hardware_profile()
print(f'Detected Profile: {profile.label}')

config.GPU_MEMORY_BUDGET = profile.gpu_memory_budget
config.CPU_MEMORY_BUDGET = profile.cpu_memory_budget
config.MAX_NATIVE_FRAMES = profile.max_native_frames

gen = VideoGenerator()

prompt = 'A beautiful golden retriever puppy playing with a red ball on lush green grass, cinematic lighting, slow motion'
negative_prompt = config.NEGATIVE_PROMPT

print('\nParameters:')
print(f' - Prompt: {prompt}')
print(' - Resolution: 384x224 (Safest RTX 4050 Preset)')
print(' - Frames: 49 (~1.6s @ 30fps)')
print(' - Steps: 15')
print(' - Guidance Scale: 3.0')

try:
    output_path = gen.generate_text_to_video(
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=384,
        height=224,
        num_frames=49,
        num_inference_steps=15,
        guidance_scale=3.0,
        seed=42,
    )
    print(f'\n[SUCCESS] Generated video saved at: {output_path}')
    print('\nGeneration QC Report:')
    print(gen.generation_report())
except Exception as e:
    print(f'\n[ERROR] Generation failed: {e}')
    import traceback
    traceback.print_exc()
