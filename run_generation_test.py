"""Test script to generate a 4-8 second video using the optimized pipeline."""
from __future__ import annotations

import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.resolve()))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import torch
import config
from engine.hardware_profiles import get_active_hardware_profile
from engine.optimized_generator import OptimizedVideoGenerator
from engine.longform import LongFormVideoGenerator, plan_story, plan_markdown
from engine.video_qc import inspect_video

def main():
    print("=" * 60)
    print("  LTX-Video: Running Output Generation Test")
    print("=" * 60)

    # 1. Hardware Profile
    profile = get_active_hardware_profile()
    print(f"[1/5] Hardware Profile Detected: {profile.label}")
    config.GPU_MEMORY_BUDGET = profile.gpu_memory_budget
    config.CPU_MEMORY_BUDGET = profile.cpu_memory_budget
    config.MAX_NATIVE_FRAMES = profile.max_native_frames
    print(f"      GPU Budget: {config.GPU_MEMORY_BUDGET} | CPU Budget: {config.CPU_MEMORY_BUDGET}")

    # 2. Plan Story for 8 seconds (2 scenes x ~4s)
    story_text = (
        "A sleek futuristic neon hovercar glides smoothly down a rainy cyberpunk boulevard at night. "
        "The vehicle accelerates into the distance as vibrant holographic neon signs reflect off the wet pavement."
    )
    duration_choice = "15 seconds" # Will create 2-3 scenes or let's target 8 seconds
    quality_choice = "Fast" # 384x224 @ 97 frames per scene, 12 steps for fast responsive verification
    aspect_choice = "YouTube / Landscape (16:9)"

    print(f"\n[2/5] Planning Story...")
    plan = plan_story(
        story=story_text,
        duration_label="15 seconds", # 2 scenes ~8s at Fast profile (97 frames each)
        quality_label=quality_choice,
        aspect_label=aspect_choice,
    )
    print(plan_markdown(plan))
    print(f"      Target: {plan.scene_count} scenes, Total ~{plan.estimated_seconds:.1f}s ({plan.width}x{plan.height})")

    # 3. Instantiate Optimized Generator & LongForm Pipeline
    print(f"\n[3/5] Initializing OptimizedVideoGenerator & LongFormVideoGenerator...")
    gen = OptimizedVideoGenerator()
    longform = LongFormVideoGenerator(gen)

    def on_progress(msg: str, val: float):
        percent = val * 100
        print(f"  --> [{percent:5.1f}%] {msg}")

    # 4. Generate Video
    print(f"\n[4/5] Starting Generation...")
    start_time = time.time()
    
    output_path = longform.generate(
        plan=plan,
        style_prompt="cinematic film lighting, photorealistic cyberpunk atmosphere, volumetric rain reflections, 8k crisp details",
        character_lock="sleek metallic cyber hovercar with glowing cyan rim lights",
        reference_image=None,
        negative_prompt=config.NEGATIVE_PROMPT,
        seed=42,
        progress_callback=on_progress,
    )
    
    elapsed = time.time() - start_time
    print(f"\n[SUCCESS] Generated video finished in {elapsed:.1f} seconds!")
    print(f"Output File: {output_path}")
    print(f"File Size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")

    # 5. Perform Quality Control Inspection
    print(f"\n[5/5] Inspecting Generated Video Quality...")
    qc = inspect_video(output_path)
    print(f"  Duration   : {qc.duration_seconds:.2f}s ({qc.frame_count} frames @ {qc.fps:.1f} fps)")
    print(f"  Resolution : {qc.width}x{qc.height}")
    print(f"  Bitrate    : {qc.bitrate_kbps:.1f} kbps")
    print(f"  Mean Brightness: {qc.mean_brightness:.1f}")
    print(f"  Contrast   : {qc.contrast:.1f}")
    print(f"  Motion Mag : {qc.motion_magnitude:.2f}")
    print(f"  Passed QC  : {qc.passed}")
    if qc.warnings:
        print(f"  Warnings   : {', '.join(qc.warnings)}")
    else:
        print("  Warnings   : None (Clean Quality Check)")

    print("\n" + "=" * 60)
    print("  Generation and Analysis Complete")
    print("=" * 60)

if __name__ == "__main__":
    main()
