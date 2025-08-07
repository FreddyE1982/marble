from __future__ import annotations

from typing import Any, Iterable, Iterator

import torch
from torch.utils.data import IterableDataset

from bit_tensor_dataset import BitTensorDataset


class BitTensorStreamingDataset(IterableDataset):
    """Stream ``(input, target)`` pairs as bit tensors with random access.

    The class is a lightweight wrapper around a HuggingFace dataset object that
    exposes ``select`` for random access. Only the requested records are pulled
    from the underlying dataset which keeps the implementation compatible with
    ``datasets.load_dataset(..., streaming=True)``.

    Parameters
    ----------
    dataset:
        HuggingFace dataset object supporting ``select``. Each sample must be a
        ``(input, target)`` tuple or a mapping with ``"input"`` and
        ``"target"`` fields.
    batch_size:
        Number of consecutive samples yielded together when iterating.
    virtual_batch_size:
        Size of the batches returned by :meth:`get_virtual_batch`. When not
        provided virtual batching is disabled.
    kwargs:
        Forwarded to :class:`BitTensorDataset` for encoding configuration.
    """

    def __init__(
        self,
        dataset: Iterable[tuple[Any, Any]],
        *,
        batch_size: int = 1,
        virtual_batch_size: int | None = None,
        **kwargs: Any,
    ) -> None:
        if not hasattr(dataset, "select"):
            raise TypeError("dataset must provide a 'select' method")
        self.dataset = dataset
        self.batch_size = int(batch_size)
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.virtual_batch_size = (
            int(virtual_batch_size) if virtual_batch_size is not None else None
        )
        if self.virtual_batch_size is not None and self.virtual_batch_size <= 0:
            raise ValueError("virtual_batch_size must be positive")
        # Reuse BitTensorDataset for encoding logic
        self.encoder = BitTensorDataset([], **kwargs)
        self.position = 0

    # ------------------------------------------------------------------
    # Internal helpers
    def _normalise_record(self, record: Any) -> tuple[Any, Any]:
        if isinstance(record, tuple):
            return record
        if isinstance(record, dict):
            return record.get("input"), record.get("target")
        raise TypeError("record must be tuple or dict with 'input'/'target'")

    def _fetch_index(self, index: int) -> tuple[Any, Any]:
        subset = self.dataset.select([index])
        try:
            record = next(iter(subset))
        except StopIteration as exc:  # dataset exhausted
            raise IndexError(index) from exc
        return self._normalise_record(record)

    # ------------------------------------------------------------------
    # Seeking
    def seek_to(self, index: int) -> None:
        if index < 0:
            raise ValueError("index must be non-negative")
        # Validate by attempting to select the record
        self._fetch_index(index)
        self.position = index

    def seek_forward(self, n: int) -> None:
        self.seek_to(self.position + int(n))

    def seek_backward(self, n: int) -> None:
        self.seek_to(max(0, self.position - int(n)))

    # ------------------------------------------------------------------
    # Iteration
    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        idx = self.position
        while True:
            inputs: list[torch.Tensor] = []
            targets: list[torch.Tensor] = []
            for _ in range(self.batch_size):
                try:
                    inp_raw, tgt_raw = self._fetch_index(idx)
                except IndexError:
                    break
                inputs.append(self.encoder.encode_object(inp_raw))
                targets.append(self.encoder.encode_object(tgt_raw))
                idx += 1
            if not inputs:
                break
            if self.batch_size == 1:
                yield inputs[0], targets[0]
            else:
                yield torch.stack(inputs), torch.stack(targets)
        self.position = idx

    # ------------------------------------------------------------------
    # Virtual batching
    def get_virtual_batch(
        self, batch_index: int, *, stream: bool = False
    ) -> Iterable[tuple[torch.Tensor, torch.Tensor]]:
        if self.virtual_batch_size is None:
            raise ValueError("virtual_batch_size not configured")
        start = batch_index * self.virtual_batch_size
        end = start + self.virtual_batch_size
        subset = self.dataset.select(range(start, end))

        def _generator() -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
            for record in subset:
                inp_raw, tgt_raw = self._normalise_record(record)
                yield (
                    self.encoder.encode_object(inp_raw),
                    self.encoder.encode_object(tgt_raw),
                )

        if stream:
            return _generator()
        return list(_generator())
