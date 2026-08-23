"""Runtime hardware profiles for laptop-class NVIDIA GPUs.

The profiles intentionally reserve VRAM for CUDA kernels, the VAE decode path,
and the desktop. They are conservative defaults, not performance guarantees.
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

    @property
    def safe_preset(self) -> str:
        return f"{self.safe_width}x{self.safe_height} · {self.safe_frames} frames"


def select_hardware_profile(
    gpu_name: str,
    vram_total_gb: float,
    ram_total_gb: float | None = None,
) -> HardwareProfile:
    """Return a conservative generation profile from GPU model and VRAM.

    RTX 3050 laptop GPUs commonly ship with either 4 GB or 6 GB VRAM. The
    4 GB path keeps roughly 1 GB of headroom and limits a single native clip
    to 121 frames; longer output should use the existing multi-scene/story
    workflow. RTX 4050 / 6 GB-class cards keep the existing 5 GiB budget.
    """
    name = (gpu_name or "unknown gpu").lower()
    ram = float(ram_total_gb or 0.0)
    cpu_budget = "8GiB" if ram >= 20 else "9GiB"

    if "3050" in name and vram_total_gb <= 4.5:
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
            cpu_memory_budget="8GiB" if ram >= 16 else "7GiB",
            max_native_frames=241,
            safe_width=384,
            safe_height=224,
            safe_frames=49,
        )

    if vram_total_gb <= 4.5:
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

    if vram_total_gb <= 6.5:
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

    return HardwareProfile(
        key="generic-8gb-plus",
        label=f"{vram_total_gb:.1f} GB VRAM profile",
        gpu_memory_budget="6GiB",
        cpu_memory_budget="8GiB",
        max_native_frames=241,
        safe_width=384,
        safe_height=224,
        safe_frames=49,
        attention_slicing=False,
    )


def get_active_hardware_profile() -> HardwareProfile:
    """Detect the current CUDA GPU lazily so pure profile tests need no GPU."""
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
