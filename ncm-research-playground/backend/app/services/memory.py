"""Per-run device memory, where the backend exposes it honestly."""
import torch

from app.schemas.generation import MemoryStats


def reset_peak(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()


def read_memory(device: str) -> MemoryStats:
    if device.startswith("cuda"):
        return MemoryStats(device=device, peak_bytes=int(torch.cuda.max_memory_allocated()),
                           note="torch.cuda.max_memory_allocated during this request")
    if device == "mps":
        return MemoryStats(device=device, peak_bytes=int(torch.mps.driver_allocated_memory()),
                           note="MPS driver allocation after the request (not a peak)")
    return MemoryStats(device=device, peak_bytes=None,
                       note="not measured: PyTorch has no per-request peak allocator stats on CPU")
