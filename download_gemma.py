import os
import sys

# Monkey-patch httpx to disable SSL verification (fixes Windows cert issues for HF Hub)
import httpx
original_init = httpx.Client.__init__
def new_init(self, *args, **kwargs):
    kwargs['verify'] = False
    original_init(self, *args, **kwargs)
httpx.Client.__init__ = new_init

from huggingface_hub import snapshot_download

def download_gemma():
    target_dir = r"E:\video\models\gemma-3-4b-it"
    os.makedirs(target_dir, exist_ok=True)
    
    print(f"Downloading unsloth/gemma-3-4b-it to {target_dir}...")
    try:
        # Using unsloth's ungated version of gemma-3-4b-it
        snapshot_download(
            repo_id="unsloth/gemma-3-4b-it",
            local_dir=target_dir,
            local_dir_use_symlinks=False,
            resume_download=True,
            ignore_patterns=["*.gguf", "*.pth"]
        )
        print("✅ Gemma Text Encoder downloaded successfully!")
    except Exception as e:
        print(f"❌ Download failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    download_gemma()
