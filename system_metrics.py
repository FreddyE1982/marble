import psutil
import torch


def get_cpu_usage() -> float:
    """Return current system-wide CPU utilisation percentage."""
    return psutil.cpu_percent(interval=None)


def get_system_memory_usage() -> float:
    """Return current system memory usage in megabytes."""
    mem = psutil.virtual_memory()
    return mem.used / (1024 ** 2)


def get_gpu_memory_usage(device: int = 0) -> float:
    """Return GPU memory usage in megabytes for ``device`` or 0.0 if CUDA unavailable."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated(device) / (1024 ** 2)
    return 0.0

def profile_resource_usage(device: int = 0) -> dict:
    """Return current CPU usage, RAM and GPU memory usage."""
    return {
        "cpu_percent": get_cpu_usage(),
        "ram_mb": get_system_memory_usage(),
        "gpu_mb": get_gpu_memory_usage(device),
    }
