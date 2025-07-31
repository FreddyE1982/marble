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
    return torch.tensor(
        [[int(bit) for bit in f"{byte:08b}"] for byte in bytes_obj], dtype=torch.uint8
    )


def tensors_to_bytes(tensor: torch.Tensor) -> bytes:
    """Convert a ``(n, 8)`` bit tensor back to bytes."""
    assert tensor.ndim == 2 and tensor.shape[1] == 8, "Tensor must have shape (n, 8)"
    byte_list = [
        int("".join(str(bit.item()) for bit in byte_bits), 2) for byte_bits in tensor
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
    max_size: int | None = None,
    start_id: int = 256,
    min_occurrence: int = 4,
) -> dict[tuple[int, ...], int]:
    """Create a vocabulary mapping frequent bit patterns to tokens.

    Parameters
    ----------
    bitstream:
        Full bitstream collected from all objects.
    min_len:
        Minimum length of bit patterns considered a ``word``.
    max_len:
        Maximum length of patterns to evaluate.
    max_size:
        Limit on the vocabulary size or ``None`` for no limit.
    start_id:
        First token value used when assigning IDs.
    min_occurrence:
        Minimum number of times a pattern must occur to be included.
    """
    counter: Counter[tuple[int, ...]] = Counter()
    for length in range(min_len, max_len + 1):
        for i in range(len(bitstream) - length + 1):
            seq = tuple(bitstream[i : i + length])
            counter[seq] += 1

    if min_occurrence > 1:
        counter = Counter({k: v for k, v in counter.items() if v >= min_occurrence})

    limit = max_size if max_size is not None else len(counter)
    best = counter.most_common(limit)
    return {pattern: i for i, (pattern, _) in enumerate(best, start=start_id)}


def encode_with_vocab(
    bitstream: list[int],
    vocab: dict[tuple[int, ...], int],
    *,
    vocab_only: bool = False,
) -> list[int]:
    """Replace known bit patterns in ``bitstream`` with vocabulary tokens.

    Parameters
    ----------
    bitstream:
        Sequence of bits to encode.
    vocab:
        Mapping of bit patterns to integer tokens.
    vocab_only:
        When ``True`` bits that are not part of any pattern are dropped.
        Otherwise the raw bit value is kept (mixed mode).
    """
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
            if not vocab_only:
                result.append(bitstream[i])
            i += 1
    return result


def decode_with_vocab(
    encoded: list[int], vocab: dict[tuple[int, ...], int]
) -> list[int]:
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

    def __init__(
        self,
        data: Iterable[tuple[Any, Any]],
        use_vocab: bool = False,
        *,
        vocab: dict[tuple[int, ...], int] | None = None,
        mixed: bool = True,
        max_vocab_size: int | None = None,
        min_word_length: int = 4,
        max_word_length: int = 8,
        min_occurrence: int = 4,
        device: str | torch.device | None = None,
    ) -> None:
        """Prepare ``(input, target)`` pairs for training.

        When ``use_vocab`` is ``True`` a shared bit-level vocabulary is built
        from all inputs and targets to reduce tensor sizes.

        Parameters
        ----------
        mixed:
            When ``True`` the encoded bitstream keeps bits that are not part of
            the vocabulary. When ``False`` only vocabulary "words" are kept.
        max_vocab_size:
            Maximum number of words stored in the vocabulary. ``None`` disables
            the limit.
        min_word_length:
            Shortest bit pattern considered when building the vocabulary.
        max_word_length:
            Longest bit pattern examined when creating vocabulary entries.
        min_occurrence:
            Minimum frequency a pattern must reach to become part of the
            vocabulary.
        device:
            Target device for stored tensors. ``None`` selects ``"cuda"`` when
            available, otherwise ``"cpu"``.
        """

        self.raw_data = list(data)
        self.use_vocab = use_vocab or vocab is not None
        self.mixed = mixed
        self.max_vocab_size = max_vocab_size
        self.min_word_length = min_word_length
        self.max_word_length = max_word_length
        self.min_occurrence = min_occurrence
        self.device = (
            torch.device("cuda")
            if device is None and torch.cuda.is_available()
            else torch.device(device or "cpu")
        )
        self.vocab: dict[tuple[int, ...], int] | None = vocab

        if self.use_vocab and self.vocab is None:
            bitstream: list[int] = []
            for inp, out in self.raw_data:
                bitstream += flatten_tensor_to_bitstream(
                    bytes_to_tensors(object_to_bytes(inp))
                )
                bitstream += flatten_tensor_to_bitstream(
                    bytes_to_tensors(object_to_bytes(out))
                )
            self.vocab = build_vocab(
                bitstream,
                min_len=self.min_word_length,
                max_len=self.max_word_length,
                max_size=self.max_vocab_size,
                min_occurrence=self.min_occurrence,
            )

        self.data: list[tuple[torch.Tensor, torch.Tensor]] = []
        for inp, out in self.raw_data:
            in_tensor = self._obj_to_tensor(inp)
            out_tensor = self._obj_to_tensor(out)
            self.data.append((in_tensor, out_tensor))

    def _obj_to_tensor(self, obj: Any) -> torch.Tensor:
        byte_data = object_to_bytes(obj)
        bit_tensor = bytes_to_tensors(byte_data).to(self.device)
        if self.vocab is None:
            return bit_tensor
        bitstream = flatten_tensor_to_bitstream(bit_tensor)
        encoded = encode_with_vocab(
            bitstream,
            self.vocab,
            vocab_only=not self.mixed,
        )
        return torch.tensor(encoded, dtype=torch.int32, device=self.device).unsqueeze(1)

    def __len__(self) -> int:  # pragma: no cover - simple
        return len(self.data)

    def __getitem__(self, idx: int):
        return self.data[idx]

    def tensor_to_object(self, tensor: torch.Tensor) -> Any:
        cpu_tensor = tensor.to("cpu")
        if self.vocab is None:
            bit_tensor = cpu_tensor
        else:
            decoded = decode_with_vocab(cpu_tensor.squeeze(1).tolist(), self.vocab)
            bit_tensor = unflatten_bitstream_to_tensor(decoded)
        byte_data = tensors_to_bytes(bit_tensor)
        return bytes_to_object(byte_data)

    def encode_object(self, obj: Any) -> torch.Tensor:
        """Return tensor representation of ``obj`` using the dataset vocabulary."""
        return self._obj_to_tensor(obj)

    def decode_tensor(self, tensor: torch.Tensor) -> Any:
        """Inverse of :meth:`encode_object`."""
        return self.tensor_to_object(tensor)

    def get_vocab(self):
        return self.vocab

    def vocab_size(self) -> int:
        return len(self.vocab) if self.vocab is not None else 0

    def to(self, device: str | torch.device) -> "BitTensorDataset":
        """Move all stored tensors to ``device`` and return ``self``."""
        self.device = torch.device(device)
        self.data = [(a.to(self.device), b.to(self.device)) for a, b in self.data]
        return self
