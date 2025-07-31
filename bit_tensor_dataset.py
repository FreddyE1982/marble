"""Dataset that stores arbitrary objects as bit tensors."""

from __future__ import annotations

import pickle
from collections import Counter
from typing import Any, Iterable

import torch
from torch.utils.data import Dataset


def is_pickleable(obj: Any) -> bool:
    """Return ``True`` if ``obj`` can be pickled."""
    try:
        pickle.dumps(obj)
        return True
    except Exception:
        return False


def object_to_bytes(obj: Any) -> bytes:
    """Serialize ``obj`` to bytes using pickle."""
    if not is_pickleable(obj):
        raise TypeError(f"Object of type {type(obj)} is not pickleable.")
    return pickle.dumps(obj)


def bytes_to_object(b: bytes) -> Any:
    """Deserialize bytes previously produced by :func:`object_to_bytes`."""
    return pickle.loads(b)


def bytes_to_tensors(bytes_obj: bytes) -> torch.Tensor:
    """Convert bytes to a ``(n, 8)`` uint8 tensor representing bits."""
    return torch.tensor([
        [int(bit) for bit in f"{byte:08b}"]
        for byte in bytes_obj
    ], dtype=torch.uint8)


def tensors_to_bytes(tensor: torch.Tensor) -> bytes:
    """Convert a ``(n, 8)`` bit tensor back to bytes."""
    assert tensor.ndim == 2 and tensor.shape[1] == 8, "Tensor must have shape (n, 8)"
    byte_list = [
        int("".join(str(bit.item()) for bit in byte_bits), 2)
        for byte_bits in tensor
    ]
    return bytes(byte_list)


def flatten_tensor_to_bitstream(tensor: torch.Tensor) -> list[int]:
    """Return the flattened bitstream of ``tensor`` as a Python list."""
    return tensor.flatten().tolist()


def unflatten_bitstream_to_tensor(bitstream: list[int]) -> torch.Tensor:
    """Convert a flat bitstream back to a ``(n, 8)`` tensor."""
    padding = (8 - len(bitstream) % 8) % 8
    bitstream += [0] * padding
    return torch.tensor(bitstream, dtype=torch.uint8).view(-1, 8)


def build_vocab(
    bitstream: list[int],
    min_len: int = 3,
    max_len: int = 8,
    top_k: int = 256,
    start_id: int = 256,
) -> dict[tuple[int, ...], int]:
    """Create a vocabulary mapping frequent bit patterns to tokens."""
    counter: Counter[tuple[int, ...]] = Counter()
    for length in range(min_len, max_len + 1):
        for i in range(len(bitstream) - length + 1):
            seq = tuple(bitstream[i : i + length])
            counter[seq] += 1
    best = counter.most_common(top_k)
    return {pattern: i for i, (pattern, _) in enumerate(best, start=start_id)}


def encode_with_vocab(bitstream: list[int], vocab: dict[tuple[int, ...], int]) -> list[int]:
    """Replace known bit patterns in ``bitstream`` with vocabulary tokens."""
    i = 0
    result: list[int] = []
    max_len = max(len(k) for k in vocab.keys()) if vocab else 0
    while i < len(bitstream):
        match = None
        for length in reversed(range(2, max_len + 1)):
            if i + length <= len(bitstream):
                candidate = tuple(bitstream[i : i + length])
                if candidate in vocab:
                    result.append(vocab[candidate])
                    i += length
                    match = True
                    break
        if not match:
            result.append(bitstream[i])
            i += 1
    return result


def decode_with_vocab(encoded: list[int], vocab: dict[tuple[int, ...], int]) -> list[int]:
    """Expand vocabulary tokens back into bit patterns."""
    inverse = {v: list(k) for k, v in vocab.items()}
    result: list[int] = []
    for token in encoded:
        if token in inverse:
            result.extend(inverse[token])
        else:
            result.append(token)
    return result


class BitTensorDataset(Dataset):
    """Dataset storing pairs as bit tensors with optional vocabulary encoding."""

    def __init__(self, data: Iterable[tuple[Any, Any]], use_vocab: bool = False) -> None:
        """Prepare ``(input, target)`` pairs for training.

        When ``use_vocab`` is ``True`` a shared bit-level vocabulary is built
        from all inputs and targets to reduce tensor sizes.
        """

        self.raw_data = list(data)
        self.use_vocab = use_vocab
        self.vocab: dict[tuple[int, ...], int] | None = None

        if self.use_vocab:
            bitstream: list[int] = []
            for inp, out in self.raw_data:
                bitstream += flatten_tensor_to_bitstream(bytes_to_tensors(object_to_bytes(inp)))
                bitstream += flatten_tensor_to_bitstream(bytes_to_tensors(object_to_bytes(out)))
            self.vocab = build_vocab(bitstream)

        self.data: list[tuple[torch.Tensor, torch.Tensor]] = []
        for inp, out in self.raw_data:
            in_tensor = self._obj_to_tensor(inp)
            out_tensor = self._obj_to_tensor(out)
            self.data.append((in_tensor, out_tensor))

    def _obj_to_tensor(self, obj: Any) -> torch.Tensor:
        byte_data = object_to_bytes(obj)
        bit_tensor = bytes_to_tensors(byte_data)
        if self.vocab is None:
            return bit_tensor
        bitstream = flatten_tensor_to_bitstream(bit_tensor)
        encoded = encode_with_vocab(bitstream, self.vocab)
        return torch.tensor(encoded, dtype=torch.int32).unsqueeze(1)

    def __len__(self) -> int:  # pragma: no cover - simple
        return len(self.data)

    def __getitem__(self, idx: int):
        return self.data[idx]

    def tensor_to_object(self, tensor: torch.Tensor) -> Any:
        if self.vocab is None:
            bit_tensor = tensor
        else:
            decoded = decode_with_vocab(tensor.squeeze(1).tolist(), self.vocab)
            bit_tensor = unflatten_bitstream_to_tensor(decoded)
        byte_data = tensors_to_bytes(bit_tensor)
        return bytes_to_object(byte_data)

    def get_vocab(self):
        return self.vocab
