from __future__ import annotations

import threading
import time
from typing import Dict

import torch

INT_DTYPES = {
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.bool,
}


def compute_delta(current: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """Compute diff between ``current`` and ``reference``.

    Integer tensors use bitwise XOR while floating tensors use subtraction.
    """
    if current.dtype in INT_DTYPES:
        return torch.bitwise_xor(current, reference)
    return current - reference


def apply_delta(tensor: torch.Tensor, delta: torch.Tensor) -> None:
    """Apply ``delta`` to ``tensor`` in-place."""
    if tensor.dtype in INT_DTYPES:
        tensor.bitwise_xor_(delta)
    else:
        tensor.add_(delta)


class TensorSyncService:
    """Background synchronisation of tensors across devices.

    Each registered device spawns a worker thread that periodically computes the
    delta to the global tensor and broadcasts it to other devices. Incoming
    deltas are applied atomically. Metrics are aggregated to detect stale devices
    which are resynchronised with the full tensor when necessary.
    """

    def __init__(self, interval_ms: int = 1000):
        self.interval = interval_ms / 1000.0
        self.devices: Dict[str, _DeviceWorker] = {}
        self.global_tensor: torch.Tensor | None = None
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.metrics = {"bytes_sent": 0, "syncs": 0, "stale_resyncs": 0}

    def register(self, name: str, tensor: torch.Tensor) -> None:
        if self.global_tensor is None:
            self.global_tensor = tensor.detach().cpu().clone()
        worker = _DeviceWorker(self, name, tensor)
        self.devices[name] = worker
        worker.start()

    def stop(self) -> None:
        self.stop_event.set()
        for w in self.devices.values():
            w.join()

    # internal helpers -------------------------------------------------

    def _broadcast(self, sender: str, delta: torch.Tensor) -> None:
        for name, worker in self.devices.items():
            if name == sender:
                continue
            worker.apply_delta(delta.to(worker.tensor.device))
        self.metrics["syncs"] += 1
        non_zero = int(torch.count_nonzero(delta))
        self.metrics["bytes_sent"] += delta.element_size() * non_zero
        self._check_stale()

    def _check_stale(self) -> None:
        if self.global_tensor is None:
            return
        now = time.time()
        size = self.global_tensor.element_size() * self.global_tensor.nelement()
        for worker in self.devices.values():
            if now - worker.last_sync > self.interval * 3:
                worker.resync(self.global_tensor)
                self.metrics["bytes_sent"] += size
                self.metrics["stale_resyncs"] += 1


class _DeviceWorker(threading.Thread):
    def __init__(self, service: TensorSyncService, name: str, tensor: torch.Tensor):
        super().__init__(daemon=True)
        self.service = service
        self.name = name
        self.tensor = tensor
        self.lock = threading.Lock()
        self.last_sync = time.time()

    def run(self) -> None:
        while not self.service.stop_event.wait(self.service.interval):
            with self.service.lock:
                global_ref = self.service.global_tensor.to(self.tensor.device)
                delta = compute_delta(self.tensor, global_ref)
                if torch.any(delta):
                    apply_delta(
                        self.service.global_tensor,
                        delta.to(self.service.global_tensor.device),
                    )
                    self.service._broadcast(self.name, delta)
                    self.last_sync = time.time()

    def apply_delta(self, delta: torch.Tensor) -> None:
        with self.lock:
            apply_delta(self.tensor, delta)
            self.last_sync = time.time()

    def resync(self, global_tensor: torch.Tensor) -> None:
        with self.lock:
            self.tensor.copy_(global_tensor.to(self.tensor.device))
            self.last_sync = time.time()
