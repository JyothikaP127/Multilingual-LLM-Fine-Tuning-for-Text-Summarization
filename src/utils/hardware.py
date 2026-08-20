"""Hardware and precision auto-detection.

No machine-specific assumptions or hardcoded paths/devices live here -- every
value is discovered at runtime so the same code works on the CPU-only local
dev machine and on whatever cloud GPU (Colab/Kaggle) ends up running training.
"""
from __future__ import annotations

import os
import platform
from dataclasses import asdict, dataclass


@dataclass
class HardwareProfile:
    platform: str
    python_version: str
    cpu_count: int
    device: str
    cuda_available: bool
    gpu_name: str | None
    gpu_total_memory_gb: float | None
    cuda_bf16_supported: bool
    torch_version: str

    def to_dict(self) -> dict:
        return asdict(self)


def detect_hardware() -> HardwareProfile:
    import torch

    cuda_available = torch.cuda.is_available()
    gpu_name = None
    gpu_mem_gb = None
    bf16_supported = False

    if cuda_available:
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        bf16_supported = torch.cuda.is_bf16_supported()

    return HardwareProfile(
        platform=platform.platform(),
        python_version=platform.python_version(),
        cpu_count=os.cpu_count() or 1,
        device="cuda" if cuda_available else "cpu",
        cuda_available=cuda_available,
        gpu_name=gpu_name,
        gpu_total_memory_gb=gpu_mem_gb,
        cuda_bf16_supported=bf16_supported,
        torch_version=torch.__version__,
    )


def resolve_precision(requested: str, profile: HardwareProfile) -> str:
    """Resolve a config precision value ('auto' | 'fp32' | 'bf16' | 'fp16').

    'auto' picks bf16 only on a CUDA GPU that supports it, otherwise fp32.
    fp16 is never chosen automatically: mT5 is documented to produce NaN
    losses under fp16 mixed precision (it was pretrained in bf16). Callers
    can still force fp16 explicitly via config if they accept that risk.
    """
    if requested != "auto":
        return requested
    if profile.cuda_available and profile.cuda_bf16_supported:
        return "bf16"
    return "fp32"
