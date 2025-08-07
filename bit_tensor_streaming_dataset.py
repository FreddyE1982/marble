from __future__ import annotations

from typing import Any, Iterable, Iterator

import torch
from torch.utils.data import IterableDataset

from bit_tensor_dataset import BitTensorDataset


class BitTensorStreamingDataset(IterableDataset):
    """Stream ``(input, target)`` pairs as bit tensors.

    This dataset converts each item from ``data_stream`` to the bit-tensor
    representation used by :class:`BitTensorDataset` and yields encoded pairs
    on-the-fly. Unlike :class:`BitTensorDataset` no data is stored in memory;
    each record is processed lazily as the stream is consumed. When
    ``batch_size`` is greater than ``1`` the dataset yields stacked batches of
    that size. The final batch may contain fewer samples if the stream is not a
    multiple of ``batch_size``.

    Parameters
    ----------
    data_stream:
        Iterable producing ``(input, target)`` pairs.
    batch_size:
        Number of samples yielded together. ``1`` streams single records.
    All additional keyword arguments are forwarded to :class:`BitTensorDataset`
    to control vocabulary, compression and device placement.
    """

    def __init__(
        self,
        data_stream: Iterable[tuple[Any, Any]],
        *,
        batch_size: int = 1,
        **kwargs: Any,
    ) -> None:
        self.data_stream = data_stream
        self.batch_size = int(batch_size)
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        # Reuse BitTensorDataset for encoding logic
        self.encoder = BitTensorDataset([], **kwargs)

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        it = iter(self.data_stream)
        if self.batch_size == 1:
            for inp, tgt in it:
                yield self.encoder.encode_object(inp), self.encoder.encode_object(tgt)
        else:
            inputs: list[torch.Tensor] = []
            targets: list[torch.Tensor] = []
            for inp, tgt in it:
                inputs.append(self.encoder.encode_object(inp))
                targets.append(self.encoder.encode_object(tgt))
                if len(inputs) == self.batch_size:
                    yield torch.stack(inputs), torch.stack(targets)
                    inputs.clear()
                    targets.clear()
            if inputs:
                yield torch.stack(inputs), torch.stack(targets)
