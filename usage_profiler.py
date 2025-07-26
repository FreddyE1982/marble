import os
import time
from dataclasses import dataclass
from typing import List

from system_metrics import get_cpu_usage, get_system_memory_usage, get_gpu_memory_usage


@dataclass
class UsageRecord:
    epoch: int
    wall_time: float
    cpu_usage: float
    ram_usage: float
    gpu_usage: float


class UsageProfiler:
    """Record CPU/GPU utilisation and runtime for training epochs."""

    def __init__(self, log_path: str, interval: int = 1) -> None:
        self.log_path = log_path
        self.interval = max(1, int(interval))
        self._start: float | None = None
        self.records: List[UsageRecord] = []
        if log_path:
            dir_name = os.path.dirname(log_path)
            if dir_name:
                os.makedirs(dir_name, exist_ok=True)
            if not os.path.exists(log_path):
                with open(log_path, "w", encoding="utf-8") as f:
                    f.write("epoch,wall_time,cpu_usage,ram_usage,gpu_usage\n")

    def start_epoch(self) -> None:
        self._start = time.time()

    def log_epoch(self, epoch: int) -> None:
        if self._start is None:
            return
        elapsed = time.time() - self._start
        record = UsageRecord(
            epoch=epoch,
            wall_time=elapsed,
            cpu_usage=get_cpu_usage(),
            ram_usage=get_system_memory_usage(),
            gpu_usage=get_gpu_memory_usage(),
        )
        self.records.append(record)
        if self.log_path:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(
                    f"{record.epoch},{record.wall_time:.4f},{record.cpu_usage:.2f},{record.ram_usage:.2f},{record.gpu_usage:.2f}\n"
                )
        self._start = None
