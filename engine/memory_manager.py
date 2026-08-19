"""Hardware detection and safety guidance for low-memory LTX generation."""
from __future__ import annotations

import gc
import os
import platform
from dataclasses import dataclass

import psutil
import torch


@dataclass
class GPUInfo:
    name: str
    vram_total_gb: float
    vram_used_gb: float
    vram_free_gb: float
    cuda_version: str


@dataclass
class SystemInfo:
    ram_total_gb: float
    ram_available_gb: float
    cpu_threads: int
    os_name: str
    python_version: str
    gpu: GPUInfo | None


def get_gpu_info() -> GPUInfo | None:
    if not torch.cuda.is_available():
        return None
    props = torch.cuda.get_device_properties(0)
    allocated = torch.cuda.memory_allocated(0) / 1024**3
    total = props.total_memory / 1024**3
    return GPUInfo(props.name, total, allocated, max(0.0, total - allocated), torch.version.cuda or "N/A")


def get_system_info() -> SystemInfo:
    ram = psutil.virtual_memory()
    return SystemInfo(ram.total / 1024**3, ram.available / 1024**3, os.cpu_count() or 1, f"{platform.system()} {platform.release()}", platform.python_version(), get_gpu_info())


def clear_gpu_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def get_hardware_recommendation() -> str:
    info = get_system_info()
    if not info.gpu:
        return "❌ CUDA GPU not detected."
    tags = []
    if "4050" in info.gpu.name.lower() or info.gpu.vram_total_gb <= 6.5:
        tags.append("RTX 4050 low-VRAM profile active")
    else:
        tags.append(f"{info.gpu.vram_total_gb:.1f} GB VRAM detected")
    if info.ram_total_gb < 20:
        tags.append("16 GB RAM-safe budgets enabled")
    tags.append("8-bit LTX + CPU/GPU balancing")
    return " · ".join(tags)


def get_status_markdown() -> str:
    info = get_system_info()
    gpu = f"**{info.gpu.name}** · {info.gpu.vram_total_gb:.1f} GB VRAM · CUDA {info.gpu.cuda_version}" if info.gpu else "**No CUDA GPU detected**"
    return f"### 🖥️ Hardware\n{gpu}\n\n**RAM:** {info.ram_total_gb:.1f} GB total / {info.ram_available_gb:.1f} GB available  \n**CPU threads:** {info.cpu_threads}  \n**Profile:** {get_hardware_recommendation()}"


def print_system_report() -> None:
    print(get_status_markdown().replace("**", ""))
