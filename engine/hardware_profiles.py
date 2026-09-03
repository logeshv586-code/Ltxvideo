"""Adaptive runtime hardware profiles for NVIDIA GPUs.

Profiles are selected from the actual GPU model, per-GPU VRAM and system RAM.
The values deliberately keep headroom for CUDA kernels, VAE decoding and the
OS. Multi-GPU scheduling is handled by ``engine.optimized_generator``; VRAM is
never incorrectly added together as if two cards were one larger GPU.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class HardwareProfile:
    key: str
    label: str
    gpu_memory_budget: str
    cpu_memory_budget: str
    max_native_frames: int
    safe_width: int
    safe_height: int
    safe_frames: int
    attention_slicing: bool = True
    native_scale: float = 1.0
    long_clip_scale: float = 1.0

    @property
    def safe_preset(self) -> str:
        return f"{self.safe_width}x{self.safe_height} · {self.safe_frames} frames"


def _cpu_budget(ram_total_gb: float | None) -> str:
    """Reserve enough system RAM for Python/Gradio while allowing CPU offload."""
    ram = float(ram_total_gb or 0.0)
    if ram >= 96:
        return "48GiB"
    if ram >= 64:
        return "32GiB"
    if ram >= 48:
        return "24GiB"
    if ram >= 32:
        return "18GiB"
    if ram >= 24:
        return "14GiB"
    if ram >= 16:
        return "9GiB"
    return "7GiB"


def select_hardware_profile(
    gpu_name: str,
    vram_total_gb: float,
    ram_total_gb: float | None = None,
) -> HardwareProfile:
    """Return a conservative quality/memory profile for one physical GPU.

    ``native_scale`` is applied to normal ~4 second renders by the optimized
    generator. ``long_clip_scale`` is used for heavier 8 second renders. This
    lets a 16 GB server GPU produce more native detail than a 4-6 GB laptop
    without changing the customer-facing UI.
    """
    name = (gpu_name or "unknown gpu").lower()
    vram = max(0.0, float(vram_total_gb or 0.0))
    cpu_budget = _cpu_budget(ram_total_gb)

    if "t4" in name and vram >= 14.0:
        return HardwareProfile(
            key="t4-16gb",
            label="NVIDIA T4 16 GB server profile",
            gpu_memory_budget="14GiB",
            cpu_memory_budget=cpu_budget,
            max_native_frames=241,
            safe_width=512,
            safe_height=288,
            safe_frames=97,
            attention_slicing=False,
            native_scale=1.33,
            long_clip_scale=1.15,
        )

    if "3050" in name and vram <= 4.5:
        return HardwareProfile(
            key="rtx3050-4gb",
            label="RTX 3050 4 GB safe profile",
            gpu_memory_budget="3GiB",
            cpu_memory_budget=cpu_budget,
            max_native_frames=121,
            safe_width=384,
            safe_height=224,
            safe_frames=49,
        )

    if "3050" in name:
        return HardwareProfile(
            key="rtx3050-6gb",
            label="RTX 3050 6 GB balanced profile",
            gpu_memory_budget="4GiB",
            cpu_memory_budget=cpu_budget,
            max_native_frames=193,
            safe_width=384,
            safe_height=224,
            safe_frames=49,
        )

    if "4050" in name:
        return HardwareProfile(
            key="rtx4050-6gb",
            label="RTX 4050 6 GB balanced profile",
            gpu_memory_budget="5GiB",
            cpu_memory_budget=cpu_budget,
            max_native_frames=241,
            safe_width=384,
            safe_height=224,
            safe_frames=49,
        )

    if vram >= 20.0:
        budget = max(8, min(20, int(vram) - 2))
        return HardwareProfile(
            key="generic-20gb-plus",
            label=f"{vram:.1f} GB VRAM high-quality profile",
            gpu_memory_budget=f"{budget}GiB",
            cpu_memory_budget=cpu_budget,
            max_native_frames=241,
            safe_width=576,
            safe_height=320,
            safe_frames=97,
            attention_slicing=False,
            native_scale=1.50,
            long_clip_scale=1.25,
        )

    if vram >= 14.0:
        budget = max(8, min(14, int(vram) - 2))
        return HardwareProfile(
            key="generic-16gb",
            label=f"{vram:.1f} GB VRAM server profile",
            gpu_memory_budget=f"{budget}GiB",
            cpu_memory_budget=cpu_budget,
            max_native_frames=241,
            safe_width=512,
            safe_height=288,
            safe_frames=97,
            attention_slicing=False,
            native_scale=1.30,
            long_clip_scale=1.15,
        )

    if vram >= 10.0:
        budget = max(7, min(10, int(vram) - 2))
        return HardwareProfile(
            key="generic-12gb",
            label=f"{vram:.1f} GB VRAM quality profile",
            gpu_memory_budget=f"{budget}GiB",
            cpu_memory_budget=cpu_budget,
            max_native_frames=241,
            safe_width=512,
            safe_height=288,
            safe_frames=97,
            attention_slicing=False,
            native_scale=1.18,
            long_clip_scale=1.08,
        )

    if vram >= 7.0:
        return HardwareProfile(
            key="generic-8gb-plus",
            label=f"{vram:.1f} GB VRAM profile",
            gpu_memory_budget="6GiB",
            cpu_memory_budget=cpu_budget,
            max_native_frames=241,
            safe_width=384,
            safe_height=224,
            safe_frames=49,
            attention_slicing=False,
            native_scale=1.05,
            long_clip_scale=1.0,
        )

    if vram <= 4.5:
        return HardwareProfile(
            key="generic-4gb",
            label="4 GB low-VRAM safe profile",
            gpu_memory_budget="3GiB",
            cpu_memory_budget=cpu_budget,
            max_native_frames=121,
            safe_width=384,
            safe_height=224,
            safe_frames=49,
        )

    return HardwareProfile(
        key="generic-6gb",
        label="6 GB low-VRAM balanced profile",
        gpu_memory_budget="4GiB",
        cpu_memory_budget=cpu_budget,
        max_native_frames=193,
        safe_width=384,
        safe_height=224,
        safe_frames=49,
    )


def get_active_hardware_profile() -> HardwareProfile:
    """Detect GPU 0 lazily so pure profile tests do not require CUDA."""
    try:
        import psutil
        import torch

        ram_total_gb = psutil.virtual_memory().total / 1024**3
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            return select_hardware_profile(
                props.name,
                props.total_memory / 1024**3,
                ram_total_gb,
            )
    except Exception:
        pass

    return HardwareProfile(
        key="no-cuda",
        label="CUDA GPU not detected",
        gpu_memory_budget="3GiB",
        cpu_memory_budget="8GiB",
        max_native_frames=241,
        safe_width=384,
        safe_height=224,
        safe_frames=49,
    )


def describe_cuda_hardware() -> tuple[int, list[str], float]:
    """Return ``(gpu_count, descriptions, system_ram_gb)`` for diagnostics/UI."""
    try:
        import psutil
        import torch

        ram_gb = psutil.virtual_memory().total / 1024**3
        if not torch.cuda.is_available():
            return 0, [], ram_gb
        descriptions: list[str] = []
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            descriptions.append(
                f"GPU {index}: {props.name} · {props.total_memory / 1024**3:.1f} GB VRAM"
            )
        return len(descriptions), descriptions, ram_gb
    except Exception:
        return 0, [], 0.0
