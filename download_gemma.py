"""Compatibility helper.

The RTX 4050 build no longer downloads Gemma/LTX-2.3. It uses the official
LTX-Video 2B Diffusers cache. Run this legacy command to prepare that cache.
"""
from download_models import download

if __name__ == "__main__":
    download()
