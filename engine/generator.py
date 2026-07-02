import torch
import numpy as np
from PIL import Image
import os
import gc
from pathlib import Path

# Monkey-patch httpx to disable SSL verification (fixes Windows cert issues)
import httpx
original_init = httpx.Client.__init__
def new_init(self, *args, **kwargs):
    kwargs['verify'] = False
    original_init(self, *args, **kwargs)
httpx.Client.__init__ = new_init

try:
    from ltx_pipelines.ti2vid_two_stages import TI2VidTwoStagesPipeline
    from ltx_core.loader import LoraPathStrengthAndSDOps, LTXV_LORA_COMFY_RENAMING_MAP
    from ltx_pipelines.utils.quantization_factory import QuantizationKind
    from ltx_pipelines.utils.args import ImageConditioningInput
    from ltx_core.components.guiders import MultiModalGuiderParams
    from ltx_pipelines.utils.types import OffloadMode
    LTX_PIPELINES_AVAILABLE = True
except ImportError:
    LTX_PIPELINES_AVAILABLE = False

from config import (
    MODELS_DIR, 
    DEFAULT_RESOLUTION, 
    DEFAULT_DURATION
)
from engine.memory_manager import clear_gpu_memory, setup_memory_optimizations

MAX_VRAM_GB = 6

class VideoGenerator:
    """Handles the core DiT video generation logic using official LTX-2.3 pipelines."""
    
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pipe = None
        self.is_loaded = False
        
        # Apply initial memory optimizations for PyTorch
        setup_memory_optimizations()
        
    def _report_progress(self, msg: str, progress: float, callback=None):
        print(f"[{progress*100:.0f}%] {msg}")
        if callback:
            callback(msg, progress)

    def load_model(self, progress_callback=None):
        """Loads the official LTX-2.3 pipeline with memory offloading."""
        if self.is_loaded:
            return
            
        if not LTX_PIPELINES_AVAILABLE:
            raise RuntimeError("ltx-pipelines is not installed. Please install from Lightricks/LTX-2 repo.")

        self._report_progress("Initializing LTX-2.3 Two-Stage Pipeline...", 0.1, progress_callback)
        
        try:
            # Paths to downloaded models
            base_model_path = os.path.join(MODELS_DIR, "ltx-2.3-22b-distilled-1.1.safetensors")
            lora_path = os.path.join(MODELS_DIR, "ltx-2.3-22b-distilled-lora-384-1.1.safetensors")
            upscaler_path = os.path.join(MODELS_DIR, "ltx-2.3-spatial-upscaler-x2-1.1.safetensors")
            gemma_path = os.path.join(MODELS_DIR, "gemma-3-4b-it")
            
            if not os.path.exists(gemma_path):
                raise FileNotFoundError("Gemma Text Encoder is missing. Please download google/gemma-3-4b-it to models/gemma-3-4b-it to run the model.")

            self._report_progress("Loading model weights (this takes a while)...", 0.3, progress_callback)
            
            # Configure heavy offloading for 6GB VRAM
            offload_mode = OffloadMode.CPU if MAX_VRAM_GB <= 8 else OffloadMode.NONE
            
            self.pipe = TI2VidTwoStagesPipeline(
                checkpoint_path=base_model_path,
                distilled_lora=[LoraPathStrengthAndSDOps(path=lora_path, strength=1.0, sd_ops=LTXV_LORA_COMFY_RENAMING_MAP)],
                spatial_upsampler_path=upscaler_path,
                gemma_root=gemma_path,
                loras=[],
                device=torch.device(self.device),
                quantization=QuantizationKind("fp8-cast").to_policy(checkpoint_path=base_model_path),
                offload_mode=offload_mode,
            )
            
            self.is_loaded = True
            self._report_progress("Model loaded successfully!", 1.0, progress_callback)
            
        except Exception as e:
            self.is_loaded = False
            error_msg = str(e)
            print(f"❌ Failed to load model: {error_msg}")
            raise RuntimeError(f"Model loading failed: {error_msg}") from e

    def unload_model(self):
        """Aggressively unloads the model to free VRAM."""
        if self.pipe is not None:
            del self.pipe
            self.pipe = None
        self.is_loaded = False
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def generate_text_to_video(
        self,
        prompt: str,
        negative_prompt: str = "worst quality, inconsistent, blurry, deformed",
        width: int = 512,
        height: int = 288,
        num_frames: int = 97,
        num_inference_steps: int = 8,
        guidance_scale: float = 3.0,
        seed: int = -1,
        progress_callback=None
    ) -> list[np.ndarray]:
        """Generates video from text using LTX-2.3."""
        
        if seed == -1:
            seed = torch.randint(0, 2**32 - 1, (1,)).item()
            
        print(f"Starting generation with seed {seed}")
        
        self.load_model(progress_callback)
        
        self._report_progress("Preparing generation...", 0.1, progress_callback)
        clear_gpu_memory()
        
        try:
            # Prepare guider params
            video_guider = MultiModalGuiderParams(
                guidance_scale=guidance_scale,
                guidance_scale_start=guidance_scale,
                guidance_scale_end=guidance_scale,
            )
            audio_guider = MultiModalGuiderParams(
                guidance_scale=1.0,
                guidance_scale_start=1.0,
                guidance_scale_end=1.0,
            )
            
            self._report_progress("Denoising...", 0.3, progress_callback)
            
            # Generate!
            video_iterator, audio = self.pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                seed=seed,
                height=height,
                width=width,
                num_frames=num_frames,
                frame_rate=24.0,
                num_inference_steps=num_inference_steps,
                video_guider_params=video_guider,
                audio_guider_params=audio_guider,
                images=[],
                enhance_prompt=False,
            )
            
            # Consume the iterator to get the final video tensor
            final_video = None
            step = 0
            for output in video_iterator:
                final_video = output
                step += 1
                progress = 0.3 + (0.6 * (step / (num_inference_steps * 2))) # rough estimate for 2 stages
                self._report_progress(f"Denoising step {step}...", progress, progress_callback)
                
            self._report_progress("Decoding frames...", 0.9, progress_callback)
            
            # Convert final video tensor to numpy frames [F, H, W, C]
            video_np = final_video.cpu().numpy()
            
            # Unload model if VRAM is severely constrained
            if MAX_VRAM_GB <= 8:
                self.unload_model()
                
            self._report_progress("Generation complete!", 1.0, progress_callback)
            return video_np
            
        except Exception as e:
            print(f"❌ Generation failed: {str(e)}")
            raise

    def test(self):
        """Simple integration test for the generator."""
        print("Running integration test...")
        frames = self.generate_text_to_video(
            prompt="A blue sphere rotating on a white background",
            width=256,
            height=256,
            num_frames=49,
            num_inference_steps=2,
            seed=42
        )
        print(f"Successfully generated {len(frames)} frames with shape {frames[0].shape}")
        
if __name__ == "__main__":
    g = VideoGenerator()
    g.test()
