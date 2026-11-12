"""CPU / GPU memory monitoring."""
from __future__ import annotations

import os
import platform

import psutil
import torch


def system_info() -> dict:
    vm = psutil.virtual_memory()
    proc = psutil.Process(os.getpid())
    info: dict = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cpu": {
            "cores": os.cpu_count(),
            "torch_threads": torch.get_num_threads(),
            "ram_total": vm.total,
            "ram_available": vm.available,
            "ram_used_percent": vm.percent,
            "process_rss": proc.memory_info().rss,
        },
        "cuda": {"available": torch.cuda.is_available(), "devices": []},
    }
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            free, total = torch.cuda.mem_get_info(i)
            info["cuda"]["devices"].append(
                {
                    "index": i,
                    "name": torch.cuda.get_device_name(i),
                    "total": total,
                    "free": free,
                    "allocated": torch.cuda.memory_allocated(i),
                    "reserved": torch.cuda.memory_reserved(i),
                }
            )
    else:
        info["cuda"]["note"] = "No CUDA device visible to PyTorch; all computation runs on CPU."
    return info
