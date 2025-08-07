from __future__ import annotations

import os
import torch
import torch.multiprocessing as mp
from typing import Callable, Iterable, List


class SharedDataset:
    """Dataset that stores tensors in shared memory for cross-process access.

    ``torch.Tensor.share_memory_`` is used for CPU tensors which creates a
    memory map so that workers spawned via ``torch.multiprocessing`` can read
    the data without additional copies.  GPU tensors are simply moved to the
    selected CUDA device where they can be referenced by multiple processes
    through CUDA's IPC mechanisms.
    """

    def __init__(self, tensors: List[torch.Tensor]):
        self._tensors = tensors

    @classmethod
    def from_tensors(
        cls, tensors: Iterable[torch.Tensor], device: str | None = None
    ) -> "SharedDataset":
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        shared: List[torch.Tensor] = []
        for t in tensors:
            tensor = t.to(device)
            if device == "cpu":
                tensor = tensor.clone().share_memory_()
            shared.append(tensor)
        return cls(shared)

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self._tensors)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self._tensors[idx]


def _worker(
    dataset: SharedDataset,
    task: Callable[[torch.Tensor], torch.Tensor],
    idx_queue: mp.Queue,
    result_queue: mp.Queue,
    device: str,
) -> None:
    if device.startswith("cuda"):
        torch.cuda.set_device(device)
    while True:
        idx = idx_queue.get()
        if idx is None:
            break
        try:
            tensor = dataset[idx]
            if tensor.device.type != device:
                tensor = tensor.to(device)
            res = task(tensor)
            result_queue.put((idx, res))
        except Exception as exc:  # pragma: no cover - debugging aid
            result_queue.put((idx, exc))
            break


class ProcessManager:
    """Coordinate execution of a callable over dataset items across workers."""

    def __init__(
        self,
        dataset: SharedDataset,
        num_workers: int | None = None,
    ) -> None:
        env_workers = int(os.getenv("MARBLE_WORKERS", "0"))
        self.num_workers = num_workers or env_workers or mp.cpu_count()
        self.dataset = dataset

    def run(
        self,
        task: Callable[[torch.Tensor], torch.Tensor],
        device: str | None = None,
    ) -> List[torch.Tensor]:
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # ``fork`` can lead to socket deadlocks or ``ConnectionResetError`` on
        # some platforms when combined with multi-threaded parents.  Using the
        # ``spawn`` start method is slower to initialise but is the most robust
        # choice and works for both CPU and GPU execution.  We still allow
        # overriding the start method through the ``MARBLE_MP_START``
        # environment variable so advanced users can opt back into ``fork`` or
        # ``forkserver`` when desired.
        start = os.getenv("MARBLE_MP_START", "spawn")
        try:
            ctx = mp.get_context(start)
            idx_queue: mp.Queue = ctx.Queue()
            result_queue: mp.Queue = ctx.Queue()

            for idx in range(len(self.dataset)):
                idx_queue.put(idx)
            for _ in range(self.num_workers):
                idx_queue.put(None)

            workers = [
                ctx.Process(
                    target=_worker,
                    args=(self.dataset, task, idx_queue, result_queue, device),
                )
                for _ in range(self.num_workers)
            ]
            for p in workers:
                p.start()

            results: List[torch.Tensor] = [None] * len(self.dataset)  # type: ignore
            received = 0
            while received < len(self.dataset):
                idx, tensor = result_queue.get()
                if isinstance(tensor, Exception):
                    raise tensor
                results[idx] = tensor
                received += 1

            for p in workers:
                p.join()

            return results
        except Exception:
            from concurrent.futures import ThreadPoolExecutor

            def run_idx(idx: int) -> torch.Tensor:
                tensor = self.dataset[idx]
                if tensor.device.type != device:
                    tensor = tensor.to(device)
                return task(tensor)

            with ThreadPoolExecutor(max_workers=self.num_workers) as ex:
                return list(ex.map(run_idx, range(len(self.dataset))))
