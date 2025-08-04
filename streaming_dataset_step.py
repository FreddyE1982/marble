from __future__ import annotations

import asyncio
from typing import AsyncIterator

import torch

from bit_tensor_dataset import BitTensorDataset


class StreamingDatasetStep:
    """Asynchronously yield batches from a :class:`BitTensorDataset`.

    The step prefetches data batches in a background task using an
    :class:`asyncio.Queue`. Each batch is moved to the requested device
    (CPU or GPU) before being enqueued. Downstream consumers call
    :meth:`next_batch` to obtain dictionaries containing ``"inputs"`` and
    ``"targets"`` tensors. When the dataset is exhausted ``None`` is
    returned and :meth:`is_finished` starts returning ``True``.

    Parameters
    ----------
    dataset:
        Source :class:`BitTensorDataset` to stream from.
    batch_size:
        Number of samples per batch.
    prefetch:
        Maximum number of prefetched batches awaiting consumption.
    device:
        Target device. ``None`` selects ``"cuda"`` when available otherwise
        ``"cpu"``.
    """

    def __init__(
        self,
        dataset: BitTensorDataset,
        *,
        batch_size: int = 1,
        prefetch: int = 2,
        device: str | torch.device | None = None,
    ) -> None:
        self.dataset = dataset
        self.batch_size = int(batch_size)
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.device = (
            torch.device("cuda")
            if device is None and torch.cuda.is_available()
            else torch.device(device or "cpu")
        )
        self._queue: asyncio.Queue[dict[str, torch.Tensor] | None] = asyncio.Queue(
            maxsize=prefetch
        )
        self._producer: asyncio.Task | None = None
        self._finished = False

    def _start(self) -> None:
        if self._producer is None:
            self._producer = asyncio.create_task(self._produce())

    async def _produce(self) -> None:
        try:
            iterator = iter(self.dataset)
            while True:
                inputs: list[torch.Tensor] = []
                targets: list[torch.Tensor] = []
                try:
                    for _ in range(self.batch_size):
                        inp, tgt = next(iterator)
                        if self.device.type != inp.device.type:
                            inp = inp.to(self.device, non_blocking=True)
                            tgt = tgt.to(self.device, non_blocking=True)
                        inputs.append(inp)
                        targets.append(tgt)
                except StopIteration:
                    if inputs:
                        batch = {
                            "inputs": torch.stack(inputs),
                            "targets": torch.stack(targets),
                        }
                        await self._queue.put(batch)
                    break
                batch = {
                    "inputs": torch.stack(inputs),
                    "targets": torch.stack(targets),
                }
                await self._queue.put(batch)
        finally:
            self._finished = True
            await self._queue.put(None)

    async def next_batch(self) -> dict[str, torch.Tensor] | None:
        """Return next batch as a dictionary or ``None`` when done."""

        self._start()
        batch = await self._queue.get()
        self._queue.task_done()
        return batch

    def is_finished(self) -> bool:
        """Return ``True`` when all data has been consumed."""

        return self._finished and self._queue.empty()

    async def __aiter__(self) -> AsyncIterator[dict[str, torch.Tensor]]:
        while True:
            batch = await self.next_batch()
            if batch is None:
                break
            yield batch

    def close(self) -> None:
        """Cancel background producer if still running."""

        if self._producer is not None:
            self._producer.cancel()
            self._producer = None
