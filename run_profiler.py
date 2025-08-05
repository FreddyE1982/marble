import json, threading, time
from dataclasses import dataclass
from pathlib import Path
from typing import List

import torch

@dataclass
class RunRecord:
    step: str
    start: float
    end: float
    device: str

class RunProfiler:
    """Record execution order of pipeline steps."""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path else None
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self.records: List[RunRecord] = []
        self._lock = threading.Lock()
        self._current: dict[int, tuple[str, float, str]] = {}

    def start(self, step: str, device: torch.device) -> None:
        if device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()
        tid = threading.get_ident()
        with self._lock:
            self._current[tid] = (step, start, device.type)

    def end(self) -> None:
        tid = threading.get_ident()
        if tid not in self._current:
            return
        step, start, dev = self._current.pop(tid)
        if dev == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()
        end = time.perf_counter()
        with self._lock:
            self.records.append(RunRecord(step, start, end, dev))

    def save(self) -> None:
        if not self.path:
            return
        data = [r.__dict__ for r in sorted(self.records, key=lambda r: r.start)]
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
