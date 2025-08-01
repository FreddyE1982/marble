"""Dataset that stores arbitrary objects as bit tensors."""

from __future__ import annotations

import pickle
import zlib
import json
import base64
import hashlib
from collections import Counter
from typing import Any, Iterable, Callable

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
        start_id: int = 256,
        device: str | torch.device | None = None,
        compress: bool = False,
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
        start_id:
            First vocabulary token index when ``use_vocab`` is ``True``.
        compress:
            When ``True`` all objects are compressed using ``zlib`` before
            converting them to bit tensors. This can substantially reduce
            dataset size when storing large pickled objects at the cost of
            slightly longer encode/decode times.
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
        self.compress = compress
        self.vocab: dict[tuple[int, ...], int] | None = vocab
        self.start_id = start_id

        if self.use_vocab and self.vocab is None:
            bitstream: list[int] = []
            for inp, out in self.raw_data:
                in_bytes = object_to_bytes(inp)
                out_bytes = object_to_bytes(out)
                if self.compress:
                    in_bytes = zlib.compress(in_bytes)
                    out_bytes = zlib.compress(out_bytes)
                bitstream += flatten_tensor_to_bitstream(
                    bytes_to_tensors(in_bytes)
                )
                bitstream += flatten_tensor_to_bitstream(
                    bytes_to_tensors(out_bytes)
                )
            self.vocab = build_vocab(
                bitstream,
                min_len=self.min_word_length,
                max_len=self.max_word_length,
                max_size=self.max_vocab_size,
                min_occurrence=self.min_occurrence,
                start_id=self.start_id,
            )

        self.data: list[tuple[torch.Tensor, torch.Tensor]] = []
        for inp, out in self.raw_data:
            in_tensor = self._obj_to_tensor(inp)
            out_tensor = self._obj_to_tensor(out)
            self.data.append((in_tensor, out_tensor))

    def _obj_to_tensor(self, obj: Any) -> torch.Tensor:
        byte_data = object_to_bytes(obj)
        if self.compress:
            byte_data = zlib.compress(byte_data)
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
        if self.compress:
            byte_data = zlib.decompress(byte_data)
        return bytes_to_object(byte_data)

    def encode_object(self, obj: Any) -> torch.Tensor:
        """Return tensor representation of ``obj`` using the dataset vocabulary."""
        return self._obj_to_tensor(obj)

    def decode_tensor(self, tensor: torch.Tensor) -> Any:
        """Inverse of :meth:`encode_object`."""
        return self.tensor_to_object(tensor)

    def add_pair(self, inp: Any, target: Any) -> None:
        """Append a single ``(input, target)`` pair to the dataset."""
        self.raw_data.append((inp, target))
        in_tensor = self._obj_to_tensor(inp)
        out_tensor = self._obj_to_tensor(target)
        self.data.append((in_tensor, out_tensor))

    def extend(self, pairs: Iterable[tuple[Any, Any]]) -> None:
        """Add multiple pairs to the dataset."""
        for inp, target in pairs:
            self.add_pair(inp, target)

    def get_vocab(self):
        return self.vocab

    def vocab_size(self) -> int:
        return len(self.vocab) if self.vocab is not None else 0

    def to(self, device: str | torch.device) -> "BitTensorDataset":
        """Move all stored tensors to ``device`` and return ``self``."""
        self.device = torch.device(device)
        self.data = [(a.to(self.device), b.to(self.device)) for a, b in self.data]
        return self

    def __iter__(self):
        """Yield each ``(input, target)`` pair in sequence."""
        return iter(self.data)

    def iter_decoded(self) -> Iterable[tuple[Any, Any]]:
        """Yield decoded ``(input, target)`` pairs one at a time.

        This convenience iterator simplifies inspecting the stored data
        without manually converting each tensor back to its original
        Python object. It avoids allocating intermediate lists so even
        large datasets can be streamed efficiently.
        """
        for inp, out in self.data:
            yield self.tensor_to_object(inp), self.tensor_to_object(out)

    def summary(self) -> dict[str, Any]:
        """Return basic statistics about the dataset.

        The summary includes the number of stored pairs, the current
        vocabulary size and whether compression is enabled. The device the
        tensors reside on is also reported. This helper simplifies logging and
        debugging by providing a quick overview of key attributes.
        """

        total_elements = sum(
            a.numel() + b.numel() for a, b in self.data
        )
        total_bytes = sum(
            a.element_size() * a.numel() + b.element_size() * b.numel()
            for a, b in self.data
        )
        avg_len = (
            float(total_elements) / len(self.data) if self.data else 0.0
        )
        avg_bytes = (
            float(total_bytes) / len(self.data) if self.data else 0.0
        )
        return {
            "num_pairs": len(self.data),
            "vocab_size": self.vocab_size(),
            "device": str(self.device),
            "compressed": self.compress,
            "start_id": self.start_id,
            "total_elements": int(total_elements),
            "avg_pair_length": avg_len,
            "total_bytes": int(total_bytes),
            "avg_pair_bytes": avg_bytes,
        }

    def save(self, path: str) -> None:
        """Persist dataset tensors and metadata to ``path``."""
        payload = {
            "data": [(a.cpu(), b.cpu()) for a, b in self.data],
            "vocab": self.vocab,
            "mixed": self.mixed,
            "max_vocab_size": self.max_vocab_size,
            "min_word_length": self.min_word_length,
            "max_word_length": self.max_word_length,
            "min_occurrence": self.min_occurrence,
            "start_id": self.start_id,
            "device": str(self.device),
            "compress": self.compress,
        }
        torch.save(payload, path)

    @classmethod
    def load(cls, path: str, *, device: str | torch.device | None = None) -> "BitTensorDataset":
        """Load a dataset previously saved with :meth:`save`."""
        obj = torch.load(path, map_location="cpu")
        ds = cls(
            [],
            use_vocab=obj["vocab"] is not None,
            vocab=obj["vocab"],
            mixed=obj["mixed"],
            max_vocab_size=obj["max_vocab_size"],
            min_word_length=obj["min_word_length"],
            max_word_length=obj["max_word_length"],
            min_occurrence=obj["min_occurrence"],
            start_id=obj.get("start_id", 256),
            device=device or obj["device"],
            compress=obj["compress"],
        )
        ds.data = [(a.to(ds.device), b.to(ds.device)) for a, b in obj["data"]]
        return ds

    def to_dict(self) -> dict[str, Any]:
        """Return a serialisable dictionary representing this dataset."""
        encoded = []
        for inp, out in self.iter_decoded():
            in_bytes = object_to_bytes(inp)
            out_bytes = object_to_bytes(out)
            if self.compress:
                in_bytes = zlib.compress(in_bytes)
                out_bytes = zlib.compress(out_bytes)
            encoded.append(
                [base64.b64encode(in_bytes).decode("ascii"), base64.b64encode(out_bytes).decode("ascii")]
            )
        if self.vocab is not None:
            vocab = {" ".join(map(str, k)): v for k, v in self.vocab.items()}
        else:
            vocab = None
        return {
            "data": encoded,
            "vocab": vocab,
            "mixed": self.mixed,
            "max_vocab_size": self.max_vocab_size,
            "min_word_length": self.min_word_length,
            "max_word_length": self.max_word_length,
            "min_occurrence": self.min_occurrence,
            "start_id": self.start_id,
            "device": str(self.device),
            "compress": self.compress,
        }

    @classmethod
    def from_dict(cls, obj: dict[str, Any], *, device: str | torch.device | None = None) -> "BitTensorDataset":
        """Reconstruct a dataset from :meth:`to_dict` output."""
        data = []
        for enc_in, enc_out in obj["data"]:
            in_bytes = base64.b64decode(enc_in)
            out_bytes = base64.b64decode(enc_out)
            if obj.get("compress"):
                in_bytes = zlib.decompress(in_bytes)
                out_bytes = zlib.decompress(out_bytes)
            inp = bytes_to_object(in_bytes)
            out = bytes_to_object(out_bytes)
            data.append((inp, out))
        if obj.get("vocab") is not None:
            vocab = {
                tuple(map(int, k.split())): v for k, v in obj["vocab"].items()
            }
        else:
            vocab = None
        return cls(
            data,
            use_vocab=obj.get("vocab") is not None,
            vocab=vocab,
            mixed=obj.get("mixed", True),
            max_vocab_size=obj.get("max_vocab_size"),
            min_word_length=obj.get("min_word_length", 4),
            max_word_length=obj.get("max_word_length", 8),
            min_occurrence=obj.get("min_occurrence", 4),
            start_id=obj.get("start_id", 256),
            device=device or obj.get("device"),
            compress=obj.get("compress", False),
        )

    def to_json(self) -> str:
        """Serialise the dataset to a JSON string."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, json_str: str, *, device: str | torch.device | None = None) -> "BitTensorDataset":
        """Load a dataset from a JSON string produced by :meth:`to_json`."""
        obj = json.loads(json_str)
        return cls.from_dict(obj, device=device)

    def map_pairs(
        self,
        transform: Callable[[Any, Any], tuple[Any, Any]],
        *,
        rebuild_vocab: bool = False,
    ) -> None:
        """Apply ``transform`` to every stored pair and optionally rebuild vocab.

        The function ``transform`` is passed each decoded ``(input, target)``
        pair and must return a new ``(input, target)`` pair. The dataset is
        updated in-place. When ``rebuild_vocab`` is ``True`` and a vocabulary is
        used, a new vocabulary is constructed from the transformed data before
        encoding the tensors again. This ensures the encoding reflects the new
        distribution of bit patterns.
        """

        new_raw = []
        for inp, out in self.iter_decoded():
            new_raw.append(transform(inp, out))

        if rebuild_vocab and self.vocab is not None:
            bitstream: list[int] = []
            for a, b in new_raw:
                in_bytes = object_to_bytes(a)
                out_bytes = object_to_bytes(b)
                if self.compress:
                    in_bytes = zlib.compress(in_bytes)
                    out_bytes = zlib.compress(out_bytes)
                bitstream += flatten_tensor_to_bitstream(bytes_to_tensors(in_bytes))
                bitstream += flatten_tensor_to_bitstream(bytes_to_tensors(out_bytes))
            self.vocab = build_vocab(
                bitstream,
                min_len=self.min_word_length,
                max_len=self.max_word_length,
                max_size=self.max_vocab_size,
                min_occurrence=self.min_occurrence,
                start_id=self.start_id,
            )

        self.raw_data = new_raw
        self.data = [
            (self._obj_to_tensor(a), self._obj_to_tensor(b)) for a, b in new_raw
        ]

    def filter_pairs(self, predicate: Callable[[Any, Any], bool]) -> None:
        """Remove pairs for which ``predicate`` returns ``False``."""

        keep_raw = []
        keep_data = []
        for (inp, out), (t_in, t_out) in zip(self.raw_data, self.data):
            if predicate(inp, out):
                keep_raw.append((inp, out))
                keep_data.append((t_in, t_out))
        self.raw_data = keep_raw
        self.data = keep_data

    def shuffle(self, *, generator: torch.Generator | None = None) -> None:
        """Randomly shuffle dataset order in-place."""

        idx = torch.randperm(len(self.data), generator=generator).tolist()
        self.raw_data = [self.raw_data[i] for i in idx]
        self.data = [self.data[i] for i in idx]

    def split(
        self,
        ratio: float,
        *,
        shuffle: bool = True,
        generator: torch.Generator | None = None,
    ) -> tuple["BitTensorDataset", "BitTensorDataset"]:
        """Return two datasets split by ``ratio``.

        Parameters
        ----------
        ratio:
            Fraction of pairs assigned to the first dataset (``0<ratio<1``).
        shuffle:
            When ``True`` pairs are shuffled before splitting.
        generator:
            Optional ``torch.Generator`` used for deterministic shuffling.
        """

        if not 0.0 < ratio < 1.0:
            raise ValueError("ratio must be between 0 and 1")
        idx = list(range(len(self.data)))
        if shuffle:
            perm = torch.randperm(len(self.data), generator=generator).tolist()
            idx = perm
        cut = int(len(self.data) * ratio)
        first_raw = [self.raw_data[i] for i in idx[:cut]]
        second_raw = [self.raw_data[i] for i in idx[cut:]]
        return (
            BitTensorDataset(
                first_raw,
                use_vocab=self.use_vocab,
                vocab=self.vocab,
                mixed=self.mixed,
                max_vocab_size=self.max_vocab_size,
                min_word_length=self.min_word_length,
                max_word_length=self.max_word_length,
                min_occurrence=self.min_occurrence,
                start_id=self.start_id,
                device=self.device,
                compress=self.compress,
            ),
            BitTensorDataset(
                second_raw,
                use_vocab=self.use_vocab,
                vocab=self.vocab,
                mixed=self.mixed,
                max_vocab_size=self.max_vocab_size,
                min_word_length=self.min_word_length,
                max_word_length=self.max_word_length,
                min_occurrence=self.min_occurrence,
                start_id=self.start_id,
                device=self.device,
                compress=self.compress,
            ),
        )

    def hash(self) -> str:
        """Return SHA256 hash of the dataset contents."""

        digest = hashlib.sha256(self.to_json().encode("utf-8")).hexdigest()
        return digest

    @staticmethod
    def collate_fn(
        batch: list[tuple[torch.Tensor, torch.Tensor]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Pad and stack a list of dataset items into batch tensors.

        This helper allows :class:`~torch.utils.data.DataLoader` to handle
        variable-length encoded examples. Each input and target tensor is padded
        to the longest sequence in the batch using zero padding. The resulting
        tensors have shape ``(batch, max_len, features)`` where ``features`` is
        either ``8`` for raw bit streams or ``1`` for vocabulary tokens.
        """

        inputs, targets = zip(*batch)
        padded_in = torch.nn.utils.rnn.pad_sequence(inputs, batch_first=True)
        padded_out = torch.nn.utils.rnn.pad_sequence(targets, batch_first=True)
        return padded_in, padded_out
