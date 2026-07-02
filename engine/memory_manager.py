"""
LTX-2.3 Video Generation Platform — Memory Manager
GPU VRAM monitoring, optimization, and automatic configuration.
"""

import gc
import os
import platform
from dataclasses import dataclass

import psutil
import torch


@dataclass
class GPUInfo:
    """GPU hardware information."""
    name: str
    vram_total_gb: float
    vram_used_gb: float
    vram_free_gb: float
    cuda_version: str
    driver_version: str


@dataclass
class SystemInfo:
    """System hardware information."""
    ram_total_gb: float
    ram_available_gb: float
    os_name: str
    python_version: str
    gpu: GPUInfo | None


def get_gpu_info() -> GPUInfo | None:
    """Get GPU information using PyTorch CUDA."""
    if not torch.cuda.is_available():
        return None

    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    mem_allocated = torch.cuda.memory_allocated(device) / (1024**3)
    mem_total = props.total_memory / (1024**3)

    return GPUInfo(
        name=props.name,
        vram_total_gb=mem_total,
        vram_used_gb=mem_allocated,
        vram_free_gb=mem_total - mem_allocated,
        cuda_version=torch.version.cuda or "N/A",
        driver_version="N/A",
    )


def get_system_info() -> SystemInfo:
    """Get complete system information."""
    ram = psutil.virtual_memory()
    return SystemInfo(
        ram_total_gb=ram.total / (1024**3),
        ram_available_gb=ram.available / (1024**3),
        os_name=f"{platform.system()} {platform.release()}",
        python_version=platform.python_version(),
        gpu=get_gpu_info(),
    )


def get_vram_usage_str() -> str:
    """Get a formatted string of current VRAM usage."""
    if not torch.cuda.is_available():
        return "No CUDA GPU"

    allocated = torch.cuda.memory_allocated() / (1024**3)
    reserved = torch.cuda.memory_reserved() / (1024**3)
    total = torch.cuda.get_device_properties(0).total_memory / (1024**3)

    return f"{allocated:.1f}GB / {total:.1f}GB (reserved: {reserved:.1f}GB)"


def clear_gpu_memory() -> None:
    """Aggressively clear GPU memory."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()


def setup_memory_optimizations() -> None:
    """Apply system-wide memory optimizations."""
    # PyTorch CUDA memory allocation config
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Reduce memory fragmentation
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"Memory optimizations applied")
    print(f"   PYTORCH_CUDA_ALLOC_CONF={os.environ.get('PYTORCH_CUDA_ALLOC_CONF')}")


def get_optimal_dtype() -> torch.dtype:
    """Determine the optimal dtype based on GPU capabilities."""
    if not torch.cuda.is_available():
        return torch.float32

    gpu_info = get_gpu_info()
    if gpu_info is None:
        return torch.float32

    # BF16 support check (Ampere and newer)
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16

    return torch.float16


def check_pagefile_warning() -> str | None:
    """Check if system virtual memory is sufficient for CPU offloading."""
    if platform.system() != "Windows":
        return None

    swap = psutil.swap_memory()
    swap_gb = swap.total / (1024**3)

    if swap_gb < 32:
        return (
            f"⚠️  Your pagefile (virtual memory) is only {swap_gb:.0f}GB. "
            f"For CPU offloading with the 22B parameter model, you need at least 64GB. "
            f"Go to System > Advanced > Performance > Virtual Memory to increase it."
        )
    return None


def print_system_report() -> None:
    """Print a comprehensive system report."""
    info = get_system_info()

    print()
    print("╔══════════════════════════════════════════════════╗")
    print("║              System Report                       ║")
    print("╚══════════════════════════════════════════════════╝")
    print(f"  OS:        {info.os_name}")
    print(f"  Python:    {info.python_version}")
    print(f"  RAM:       {info.ram_total_gb:.1f} GB total, {info.ram_available_gb:.1f} GB available")

    if info.gpu:
        print(f"  GPU:       {info.gpu.name}")
        print(f"  VRAM:      {info.gpu.vram_total_gb:.1f} GB total, {info.gpu.vram_free_gb:.1f} GB free")
        print(f"  CUDA:      {info.gpu.cuda_version}")
    else:
        print(f"  GPU:       ❌ No CUDA GPU detected")

    # Pagefile warning
    warning = check_pagefile_warning()
    if warning:
        print(f"\n  {warning}")

    # Recommendation
    if info.gpu and info.gpu.vram_total_gb < 8:
        print(f"\n  ⚠️  Low VRAM ({info.gpu.vram_total_gb:.0f}GB). Using CPU offload + FP8 quantization.")
        print(f"     Generation will be slow (~15-30 min per 10s clip).")
    elif info.gpu and info.gpu.vram_total_gb < 16:
        print(f"\n  ⚠️  Moderate VRAM ({info.gpu.vram_total_gb:.0f}GB). Using CPU offload.")
    elif info.gpu:
        print(f"\n  ✅ Good VRAM ({info.gpu.vram_total_gb:.0f}GB). Should run well!")

    print()


if __name__ == "__main__":
    print_system_report()
