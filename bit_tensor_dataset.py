"""Dataset that stores arbitrary objects as bit tensors."""

from __future__ import annotations

import pickle
import json
import msgpack
import json
import base64
import hashlib
from collections import Counter
from typing import Any, Iterable, Callable
from dataclasses import dataclass
import mmap
import requests
import os
import aiohttp
import threading

import torch
from torch.utils.data import Dataset
from memory_pool import MemoryPool
from crypto_utils import constant_time_compare, encrypt_bytes, decrypt_bytes
from data_compressor import DataCompressor


@dataclass(slots=True)
class DatasetPair:
    """Simple container for an input and target tensor."""

    inp: torch.Tensor | None = None
    tgt: torch.Tensor | None = None
    tags: list[str] | None = None


def augment_bit_tensor(
    tensor: torch.Tensor,
    *,
    flip_probability: float = 0.0,
    noise_probability: float = 0.0,
) -> torch.Tensor:
    """Return a copy of ``tensor`` with random bit flips and noise."""

    if flip_probability <= 0.0 and noise_probability <= 0.0:
        return tensor.clone()

    rand = torch.rand_like(tensor.float())
    result = tensor.clone()
    if flip_probability > 0.0:
        flips = rand < flip_probability
        result = result ^ flips.to(dtype=result.dtype)
    if noise_probability > 0.0:
        noise_mask = (rand >= flip_probability) & (
            rand < flip_probability + noise_probability
        )
        random_bits = torch.randint(0, 2, result.shape, device=result.device, dtype=result.dtype)
        result = torch.where(noise_mask, random_bits, result)
    return result


def is_pickleable(obj: Any) -> bool:
    """Return ``True`` if ``obj`` can be pickled."""
    try:
        pickle.dumps(obj)
        return True
    except Exception:
        return False


def object_to_bytes(obj: Any, *, serializer: str = "pickle") -> bytes:
    """Serialize ``obj`` to bytes using the selected ``serializer``."""
    if serializer == "pickle":
        if not is_pickleable(obj):
            raise TypeError(f"Object of type {type(obj)} is not pickleable.")
        return pickle.dumps(obj)
    if serializer == "json":
        return json.dumps(obj).encode("utf-8")
    if serializer == "msgpack":
        return msgpack.dumps(obj)
    raise ValueError(f"unknown serializer {serializer}")


def bytes_to_object(b: bytes, *, serializer: str = "pickle") -> Any:
    """Deserialize bytes produced by :func:`object_to_bytes`."""
    if serializer == "pickle":
        return pickle.loads(b)
    if serializer == "json":
        return json.loads(b.decode("utf-8"))
    if serializer == "msgpack":
        return msgpack.loads(b)
    raise ValueError(f"unknown serializer {serializer}")


def bytes_to_tensors(bytes_obj: bytes, device: str | torch.device | None = None) -> torch.Tensor:
    """Convert bytes to a ``(n, 8)`` uint8 tensor representing bits.

    Parameters
    ----------
    bytes_obj:
        Raw bytes to convert.
    device:
        Optional device the resulting tensor should reside on. When ``None`` the
        tensor is allocated on CPU. Passing ``"cuda"`` or a CUDA device will
        place the tensor on the GPU for accelerated processing.
    """

    arr = torch.tensor(list(bytes_obj), dtype=torch.uint8, device=device)
    masks = (1 << torch.arange(7, -1, -1, dtype=torch.uint8, device=arr.device))
    bits = arr.unsqueeze(1).bitwise_and(masks).ne(0).to(torch.uint8)
    return bits


def tensors_to_bytes(tensor: torch.Tensor) -> bytes:
    """Convert a ``(n, 8)`` bit tensor back to bytes."""

    assert tensor.ndim == 2 and tensor.shape[1] == 8, "Tensor must have shape (n, 8)"
    tensor = tensor.to(torch.uint8)
    weights = 1 << torch.arange(7, -1, -1, dtype=torch.uint8, device=tensor.device)
    byte_vals = torch.sum(tensor * weights, dim=1).to(torch.uint8)
    return bytes(byte_vals.tolist())


def flatten_tensor_to_bitstream(tensor: torch.Tensor) -> list[int]:
    """Return the flattened bitstream of ``tensor`` as a Python list."""
    return tensor.flatten().tolist()


def unflatten_bitstream_to_tensor(bitstream: list[int]) -> torch.Tensor:
    """Convert a flat bitstream back to a ``(n, 8)`` tensor."""
    padding = (8 - len(bitstream) % 8) % 8
    bitstream += [0] * padding
    return torch.tensor(bitstream, dtype=torch.uint8).view(-1, 8)


def _read_stream(
    url: str,
    chunk_size: int = 8192,
    max_bytes: int | None = None,
    timeout: float | None = None,
) -> bytes:
    """Return bytes downloaded from ``url`` using streaming requests."""
    with requests.get(url, stream=True, timeout=timeout) as resp:
        resp.raise_for_status()
        data = bytearray()
        for chunk in resp.iter_content(chunk_size=chunk_size):
            if not chunk:
                continue
            data.extend(chunk)
            if max_bytes is not None and len(data) >= max_bytes:
                break
    if max_bytes is not None:
        return bytes(data[:max_bytes])
    return bytes(data)


async def _read_stream_async(
    url: str,
    chunk_size: int = 8192,
    max_bytes: int | None = None,
    timeout: float | None = None,
    session: aiohttp.ClientSession | None = None,
) -> bytes:
    """Asynchronously download bytes from ``url`` using ``aiohttp``."""
    close = False
    if session is None:
        session = aiohttp.ClientSession()
        close = True
    try:
        async with session.get(url, timeout=timeout) as resp:
            resp.raise_for_status()
            data = bytearray()
            async for chunk in resp.content.iter_chunked(chunk_size):
                if not chunk:
                    continue
                data.extend(chunk)
                if max_bytes is not None and len(data) >= max_bytes:
                    break
            if max_bytes is not None:
                return bytes(data[:max_bytes])
            return bytes(data)
    finally:
        if close:
            await session.close()


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
        compression_algorithm: str = "zlib",
        compression_level: int = 6,
        encryption_key: str | bytes | None = None,
        serializer: str = "pickle",
        tags: Iterable[list[str] | None] | None = None,
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
            When ``True`` all objects are compressed before converting them
            to bit tensors. ``compression_algorithm`` selects the
            compression backend and ``compression_level`` controls the
            compression ratio. This can substantially reduce dataset size
            when storing large pickled objects at the cost of slightly
            longer encode/decode times.
        compression_algorithm:
            Compression backend used when ``compress`` is ``True``.
            Supported options are ``"zlib"`` and ``"lzma"``.
        compression_level:
            Compression level passed to the underlying algorithm.
        encryption_key:
            Optional key used to encrypt serialized objects before they are
            converted to bit tensors. Decryption with the same key is performed
            automatically when decoding.
        """

        self.raw_data = list(data)
        self.use_vocab = use_vocab or vocab is not None
        self.mixed = mixed
        self.max_vocab_size = max_vocab_size
        self.min_word_length = min_word_length
        self.max_word_length = max_word_length
        self.min_occurrence = min_occurrence
        self.device = (
            torch.device("cuda") if device is None and torch.cuda.is_available() else torch.device(device or "cpu")
        )
        self.compress = compress
        self.compressor = DataCompressor(
            level=compression_level,
            compression_enabled=compress,
            algorithm=compression_algorithm,
        )
        self.serializer = serializer
        if encryption_key is not None and isinstance(encryption_key, str):
            encryption_key = encryption_key.encode()
        self.encryption_key: bytes | None = encryption_key
        self.encrypted = encryption_key is not None
        self.vocab: dict[tuple[int, ...], int] | None = vocab
        self.start_id = start_id

        if self.use_vocab and self.vocab is None:
            bitstream: list[int] = []
            for inp, out in self.raw_data:
                in_bytes = object_to_bytes(inp, serializer=self.serializer)
                out_bytes = object_to_bytes(out, serializer=self.serializer)
                if self.compress:
                    in_bytes = self.compressor.compress(in_bytes)
                    out_bytes = self.compressor.compress(out_bytes)
                if self.encryption_key is not None:
                    in_bytes = encrypt_bytes(in_bytes, self.encryption_key)
                    out_bytes = encrypt_bytes(out_bytes, self.encryption_key)
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

        self.pair_pool = MemoryPool(DatasetPair)
        self.data: list[DatasetPair] = []
        tag_iter = iter(tags) if tags is not None else None
        for inp, out in self.raw_data:
            pair = self.pair_pool.allocate()
            pair.inp = self._obj_to_tensor(inp)
            pair.tgt = self._obj_to_tensor(out)
            pair.tags = next(tag_iter) if tag_iter else None
            self.data.append(pair)

        self.index_pool = MemoryPool(dict)
        self.index: dict[str, int] = self.index_pool.allocate()
        self.checksums = self._build_index()

        self._history: list[list[tuple[Any, Any]]] = []
        self._history_index: int = -1
        self._snapshot()

    def _obj_to_tensor(self, obj: Any) -> torch.Tensor:
        byte_data = object_to_bytes(obj, serializer=self.serializer)
        if self.compress:
            byte_data = self.compressor.compress(byte_data)
        if self.encryption_key is not None:
            byte_data = encrypt_bytes(byte_data, self.encryption_key)
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
        pair = self.data[idx]
        return pair.inp, pair.tgt

    def tensor_to_object(self, tensor: torch.Tensor) -> Any:
        cpu_tensor = tensor.to("cpu")
        if self.vocab is None:
            bit_tensor = cpu_tensor
        else:
            decoded = decode_with_vocab(cpu_tensor.squeeze(1).tolist(), self.vocab)
            bit_tensor = unflatten_bitstream_to_tensor(decoded)
        byte_data = tensors_to_bytes(bit_tensor)
        if self.encrypted and self.encryption_key is None:
            raise ValueError("encryption_key required to decode dataset")
        if self.encryption_key is not None:
            byte_data = decrypt_bytes(byte_data, self.encryption_key)
        if self.compress:
            byte_data = self.compressor.decompress(byte_data)
        return bytes_to_object(byte_data, serializer=self.serializer)

    def encode_object(self, obj: Any) -> torch.Tensor:
        """Return tensor representation of ``obj`` using the dataset vocabulary."""
        return self._obj_to_tensor(obj)

    def decode_tensor(self, tensor: torch.Tensor) -> Any:
        """Inverse of :meth:`encode_object`."""
        return self.tensor_to_object(tensor)

    def _hash_pair(self, a: torch.Tensor, b: torch.Tensor) -> str:
        m = hashlib.sha256()
        m.update(a.to("cpu").numpy().tobytes())
        m.update(b.to("cpu").numpy().tobytes())
        return m.hexdigest()

    def _build_index(self) -> list[str]:
        self.index.clear()
        checks: list[str] = []
        for idx, pair in enumerate(self.data):
            h = self._hash_pair(pair.inp, pair.tgt)
            self.index[h] = idx
            checks.append(h)
        return checks

    def hash_pair(self, idx: int) -> str:
        pair = self.data[idx]
        return self._hash_pair(pair.inp, pair.tgt)

    def get_by_hash(self, digest: str) -> tuple[torch.Tensor, torch.Tensor]:
        pair = self.data[self.index[digest]]
        return pair.inp, pair.tgt

    def deduplicate(self) -> None:
        unique: dict[str, int] = {}
        new_raw: list[tuple[Any, Any]] = []
        new_data: list[DatasetPair] = []
        for pair_raw, pair_tensor in zip(self.raw_data, self.data):
            h = self._hash_pair(pair_tensor.inp, pair_tensor.tgt)
            if h not in unique:
                unique[h] = len(new_data)
                new_raw.append(pair_raw)
                new_data.append(pair_tensor)
        self.raw_data = new_raw
        self.data = new_data
        self.checksums = self._build_index()

    def verify_checksums(self) -> None:
        current = self._build_index()
        if len(current) != len(self.checksums):
            raise ValueError("Dataset checksum mismatch")
        for saved, curr in zip(self.checksums, current):
            if not constant_time_compare(saved, curr):
                raise ValueError("Dataset checksum mismatch")
        self.checksums = current

    def add_pair(self, inp: Any, target: Any) -> None:
        """Append a single ``(input, target)`` pair to the dataset."""
        self._add_pair_no_history(inp, target)
        self._snapshot()

    def _add_pair_no_history(self, inp: Any, target: Any) -> None:
        self.raw_data.append((inp, target))
        in_tensor = self._obj_to_tensor(inp)
        out_tensor = self._obj_to_tensor(target)
        pair = self.pair_pool.allocate()
        pair.inp = in_tensor
        pair.tgt = out_tensor
        self.data.append(pair)
        h = self._hash_pair(in_tensor, out_tensor)
        self.index[h] = len(self.data) - 1
        self.checksums.append(h)

    def extend(self, pairs: Iterable[tuple[Any, Any]]) -> None:
        """Add multiple pairs to the dataset."""
        added = False
        for inp, target in pairs:
            self._add_pair_no_history(inp, target)
            added = True
        if added:
            self._snapshot()
        if pairs:
            self._snapshot()

    def patch_pairs(self, patches: dict[int, tuple[Any, Any]]) -> None:
        """Replace existing pairs at ``indices`` with new values."""
        for idx, (inp, tgt) in patches.items():
            pair = self.data[idx]
            pair.inp = self._obj_to_tensor(inp)
            pair.tgt = self._obj_to_tensor(tgt)
            h = self._hash_pair(pair.inp, pair.tgt)
            self.index[h] = idx
            self.checksums[idx] = h
        self.checksums = self._build_index()
        if patches:
            new_raw = list(self.raw_data)
            for idx, pair in patches.items():
                new_raw[idx] = pair
            self.raw_data = new_raw
            self._snapshot()

    def adapt_vocab(self, pairs: Iterable[tuple[Any, Any]]) -> None:
        """Expand vocabulary with new patterns from ``pairs``."""
        if not self.use_vocab or self.vocab is None:
            return
        bitstream: list[int] = []
        for a, b in pairs:
            in_bytes = object_to_bytes(a, serializer=self.serializer)
            out_bytes = object_to_bytes(b, serializer=self.serializer)
            if self.compress:
                in_bytes = self.compressor.compress(in_bytes)
                out_bytes = self.compressor.compress(out_bytes)
            if self.encryption_key is not None:
                in_bytes = encrypt_bytes(in_bytes, self.encryption_key)
                out_bytes = encrypt_bytes(out_bytes, self.encryption_key)
            bitstream += flatten_tensor_to_bitstream(bytes_to_tensors(in_bytes))
            bitstream += flatten_tensor_to_bitstream(bytes_to_tensors(out_bytes))
        new_vocab = build_vocab(
            bitstream,
            min_len=self.min_word_length,
            max_len=self.max_word_length,
            start_id=self.start_id,
            min_occurrence=self.min_occurrence,
        )
        next_id = max(self.vocab.values(), default=self.start_id - 1) + 1
        for pattern in new_vocab:
            if pattern not in self.vocab:
                if self.max_vocab_size is not None and len(self.vocab) >= self.max_vocab_size:
                    break
                self.vocab[pattern] = next_id
                next_id += 1

    def append_pairs(
        self,
        pairs: Iterable[tuple[Any, Any]],
        *,
        rebuild_vocab: bool = False,
    ) -> None:
        """Append ``pairs`` and optionally rebuild the vocabulary."""

        self.raw_data.extend(pairs)
        if rebuild_vocab and self.use_vocab:
            bitstream: list[int] = []
            for a, b in self.raw_data:
                in_bytes = object_to_bytes(a, serializer=self.serializer)
                out_bytes = object_to_bytes(b, serializer=self.serializer)
                if self.compress:
                    in_bytes = self.compressor.compress(in_bytes)
                    out_bytes = self.compressor.compress(out_bytes)
                if self.encryption_key is not None:
                    in_bytes = encrypt_bytes(in_bytes, self.encryption_key)
                    out_bytes = encrypt_bytes(out_bytes, self.encryption_key)
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
            self.data = []
            for a, b in self.raw_data:
                pair = self.pair_pool.allocate()
                pair.inp = self._obj_to_tensor(a)
                pair.tgt = self._obj_to_tensor(b)
                self.data.append(pair)
            self.index_pool = MemoryPool(dict)
            self.index = self.index_pool.allocate()
            self.checksums = self._build_index()
            return
        added = False
        for inp, target in pairs:
            self._add_pair_no_history(inp, target)
            added = True
        if added:
            self._snapshot()

    def augment_bits(
        self,
        *,
        flip_probability: float = 0.0,
        noise_probability: float = 0.0,
    ) -> None:
        """Apply bit-level augmentation to all stored tensors."""

        for pair in self.data:
            pair.inp = augment_bit_tensor(
                pair.inp, flip_probability=flip_probability, noise_probability=noise_probability
            )
            pair.tgt = augment_bit_tensor(
                pair.tgt, flip_probability=flip_probability, noise_probability=noise_probability
            )
        self._snapshot()

    def add_stream_pair(
        self,
        input_url: str | None = None,
        target_url: str | None = None,
        *,
        input_processor: Callable[[bytes], Any] | None = None,
        target_processor: Callable[[bytes], Any] | None = None,
        chunk_size: int = 8192,
        max_bytes: int | None = None,
        timeout: float | None = None,
    ) -> None:
        """Download stream data from ``input_url`` and ``target_url`` and add it."""

        if input_url is None and target_url is None:
            raise ValueError("at least one of input_url or target_url must be provided")

        inp_obj = None
        tgt_obj = None
        if input_url is not None:
            data = _read_stream(input_url, chunk_size, max_bytes, timeout)
            inp_obj = input_processor(data) if input_processor else data
        if target_url is not None:
            data = _read_stream(target_url, chunk_size, max_bytes, timeout)
            tgt_obj = target_processor(data) if target_processor else data

        if inp_obj is None or tgt_obj is None:
            raise ValueError("both input and target streams must be provided")

        self.add_pair(inp_obj, tgt_obj)

    async def add_stream_pair_async(
        self,
        input_url: str | None = None,
        target_url: str | None = None,
        *,
        input_processor: Callable[[bytes], Any] | None = None,
        target_processor: Callable[[bytes], Any] | None = None,
        chunk_size: int = 8192,
        max_bytes: int | None = None,
        timeout: float | None = None,
        session: aiohttp.ClientSession | None = None,
    ) -> None:
        """Asynchronously download streams and add a pair."""

        if input_url is None and target_url is None:
            raise ValueError("at least one of input_url or target_url must be provided")

        inp_obj = None
        tgt_obj = None
        if input_url is not None:
            data = await _read_stream_async(input_url, chunk_size, max_bytes, timeout, session)
            inp_obj = input_processor(data) if input_processor else data
        if target_url is not None:
            data = await _read_stream_async(target_url, chunk_size, max_bytes, timeout, session)
            tgt_obj = target_processor(data) if target_processor else data

        if inp_obj is None or tgt_obj is None:
            raise ValueError("both input and target streams must be provided")

        self.add_pair(inp_obj, tgt_obj)

    def get_vocab(self):
        return self.vocab

    def vocab_size(self) -> int:
        return len(self.vocab) if self.vocab is not None else 0

    def to(self, device: str | torch.device) -> "BitTensorDataset":
        """Move all stored tensors to ``device`` and return ``self``."""
        self.device = torch.device(device)
        for pair in self.data:
            pair.inp = pair.inp.to(self.device)
            pair.tgt = pair.tgt.to(self.device)
        return self

    def __iter__(self):
        """Yield each ``(input, target)`` pair in sequence."""
        for pair in self.data:
            yield pair.inp, pair.tgt

    def iter_decoded(self) -> Iterable[tuple[Any, Any]]:
        """Yield decoded ``(input, target)`` pairs one at a time.

        This convenience iterator simplifies inspecting the stored data
        without manually converting each tensor back to its original
        Python object. It avoids allocating intermediate lists so even
        large datasets can be streamed efficiently.
        """
        if self.encrypted and self.encryption_key is None:
            raise ValueError("encryption_key required to decode dataset")
        for pair in self.data:
            yield self.tensor_to_object(pair.inp), self.tensor_to_object(pair.tgt)

    def summary(self) -> dict[str, Any]:
        """Return basic statistics about the dataset.

        The summary includes the number of stored pairs, the current
        vocabulary size and whether compression is enabled. The device the
        tensors reside on is also reported. This helper simplifies logging and
        debugging by providing a quick overview of key attributes.
        """

        total_elements = sum(p.inp.numel() + p.tgt.numel() for p in self.data)
        total_bytes = sum(p.inp.element_size() * p.inp.numel() + p.tgt.element_size() * p.tgt.numel() for p in self.data)
        avg_len = float(total_elements) / len(self.data) if self.data else 0.0
        avg_bytes = float(total_bytes) / len(self.data) if self.data else 0.0
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

    def release_memory(self) -> None:
        """Return all allocated pair objects to the pool."""
        for pair in self.data:
            self.pair_pool.release(pair)
        self.data.clear()


    def save_vocab(self, path: str) -> None:
        if self.vocab is None:
            raise ValueError("No vocabulary available")
        data = {" ".join(map(str, k)): v for k, v in self.vocab.items()}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load_vocab(cls, path: str) -> dict[tuple[int, ...], int]:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        return {tuple(int(x) for x in k.split()): int(v) for k, v in raw.items()}

    def save(self, path: str) -> None:
        """Persist dataset tensors and metadata to ``path``."""
        payload = {
            "data": [(p.inp.cpu(), p.tgt.cpu()) for p in self.data],
            "vocab": self.vocab,
            "mixed": self.mixed,
            "max_vocab_size": self.max_vocab_size,
            "min_word_length": self.min_word_length,
            "max_word_length": self.max_word_length,
            "min_occurrence": self.min_occurrence,
            "start_id": self.start_id,
            "device": str(self.device),
            "compress": self.compress,
            "compression_algorithm": self.compressor.algorithm,
            "compression_level": self.compressor.level,
            "encrypted": self.encrypted,
            "checksums": self.checksums,
        }
        torch.save(payload, path)

    @classmethod
    def load(
        cls,
        path: str,
        *,
        device: str | torch.device | None = None,
        encryption_key: str | bytes | None = None,
        memory_mapped: bool = False,
    ) -> "BitTensorDataset":
        """Load a dataset previously saved with :meth:`save`."""
        if memory_mapped:
            with open(path, "rb") as f:
                mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
                obj = torch.load(mm, map_location="cpu")
                mm.close()
        else:
            obj = torch.load(path, map_location="cpu")
        if obj.get("encrypted") and encryption_key is None:
            raise ValueError("encryption_key required to load dataset")
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
            compression_algorithm=obj.get("compression_algorithm", "zlib"),
            compression_level=obj.get("compression_level", 6),
            encryption_key=encryption_key if obj.get("encrypted") else None,
            serializer=obj.get("serializer", "pickle"),
        )
        ds.pair_pool = MemoryPool(DatasetPair)
        ds.data = []
        for a, b in obj["data"]:
            pair = ds.pair_pool.allocate()
            pair.inp = a.to(ds.device)
            pair.tgt = b.to(ds.device)
            ds.data.append(pair)
        ds.index_pool = MemoryPool(dict)
        ds.index = ds.index_pool.allocate()
        ds.checksums = obj.get("checksums", [])
        ds.verify_checksums()
        ds.encrypted = obj.get("encrypted", ds.encrypted)
        return ds

    def to_dict(self) -> dict[str, Any]:
        """Return a serialisable dictionary representing this dataset."""
        encoded = []
        tags = []
        for inp, out in self.iter_decoded():
            in_bytes = object_to_bytes(inp, serializer=self.serializer)
            out_bytes = object_to_bytes(out, serializer=self.serializer)
            if self.compress:
                in_bytes = self.compressor.compress(in_bytes)
                out_bytes = self.compressor.compress(out_bytes)
            if self.encryption_key is not None:
                in_bytes = encrypt_bytes(in_bytes, self.encryption_key)
                out_bytes = encrypt_bytes(out_bytes, self.encryption_key)
            encoded.append(
                [
                    base64.b64encode(in_bytes).decode("ascii"),
                    base64.b64encode(out_bytes).decode("ascii"),
                ]
            )
            tags.append(None)
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
            "compression_algorithm": self.compressor.algorithm,
            "compression_level": self.compressor.level,
            "checksums": self.checksums,
            "encrypted": self.encrypted,
            "serializer": self.serializer,
            "tags": [p.tags for p in self.data],
        }

    @classmethod
    def from_dict(
        cls,
        obj: dict[str, Any],
        *,
        device: str | torch.device | None = None,
        encryption_key: str | bytes | None = None,
    ) -> "BitTensorDataset":
        """Reconstruct a dataset from :meth:`to_dict` output."""
        data = []
        tags = obj.get("tags")
        for idx, (enc_in, enc_out) in enumerate(obj["data"]):
            in_bytes = base64.b64decode(enc_in)
            out_bytes = base64.b64decode(enc_out)
            if obj.get("compress"):
                compressor = DataCompressor(
                    level=6,
                    compression_enabled=True,
                    algorithm=obj.get("compression_algorithm", "zlib"),
                )
                in_bytes = compressor.decompress(in_bytes)
                out_bytes = compressor.decompress(out_bytes)
            if obj.get("encrypted"):
                if encryption_key is None:
                    raise ValueError("encryption_key required to load dataset")
                in_bytes = decrypt_bytes(in_bytes, encryption_key)
                out_bytes = decrypt_bytes(out_bytes, encryption_key)
            inp = bytes_to_object(in_bytes, serializer=obj.get("serializer", "pickle"))
            out = bytes_to_object(out_bytes, serializer=obj.get("serializer", "pickle"))
            data.append((inp, out))
        if obj.get("vocab") is not None:
            vocab = {tuple(map(int, k.split())): v for k, v in obj["vocab"].items()}
        else:
            vocab = None
        ds = cls(
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
            compression_algorithm=obj.get("compression_algorithm", "zlib"),
            compression_level=obj.get("compression_level", 6),
            encryption_key=encryption_key if obj.get("encrypted") else None,
            tags=tags,
        )
        ds.encrypted = obj.get("encrypted", ds.encrypted)
        ds.checksums = obj.get("checksums", [])
        ds.verify_checksums()
        return ds

    def to_json(self) -> str:
        """Serialise the dataset to a JSON string."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(
        cls,
        json_str: str,
        *,
        device: str | torch.device | None = None,
        encryption_key: str | bytes | None = None,
    ) -> "BitTensorDataset":
        """Load a dataset from a JSON string produced by :meth:`to_json`."""
        obj = json.loads(json_str)
        return cls.from_dict(obj, device=device, encryption_key=encryption_key)

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
                in_bytes = object_to_bytes(a, serializer=self.serializer)
                out_bytes = object_to_bytes(b, serializer=self.serializer)
                if self.compress:
                    in_bytes = self.compressor.compress(in_bytes)
                    out_bytes = self.compressor.compress(out_bytes)
                if self.encryption_key is not None:
                    in_bytes = encrypt_bytes(in_bytes, self.encryption_key)
                    out_bytes = encrypt_bytes(out_bytes, self.encryption_key)
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
        self.data = []
        for a, b in new_raw:
            pair = self.pair_pool.allocate()
            pair.inp = self._obj_to_tensor(a)
            pair.tgt = self._obj_to_tensor(b)
            self.data.append(pair)
        self.checksums = self._build_index()
        self._snapshot()

    def filter_pairs(self, predicate: Callable[[Any, Any], bool]) -> None:
        """Remove pairs for which ``predicate`` returns ``False``."""

        keep_raw = []
        keep_data: list[DatasetPair] = []
        for raw, pair in zip(self.raw_data, self.data):
            if predicate(raw[0], raw[1]):
                keep_raw.append(raw)
                keep_data.append(pair)
        self.raw_data = keep_raw
        self.data = keep_data
        self.checksums = self._build_index()
        self._snapshot()

    def filter_by_tag(self, tag: str) -> None:
        """Keep only pairs containing ``tag`` in their tag list."""
        keep_raw: list[tuple[Any, Any]] = []
        keep_data: list[DatasetPair] = []
        for raw, pair in zip(self.raw_data, self.data):
            if pair.tags and tag in pair.tags:
                keep_raw.append(raw)
                keep_data.append(pair)
        self.raw_data = keep_raw
        self.data = keep_data
        self.checksums = self._build_index()
        self._snapshot()

    def prune_invalid(self, validator: Callable[[Any, Any], bool] | None = None) -> int:
        """Remove pairs that fail decoding or ``validator`` check."""

        removed = 0
        validator = validator or (lambda *_: True)
        keep_raw: list[tuple[Any, Any]] = []
        keep_data: list[DatasetPair] = []
        for raw, pair in zip(self.raw_data, self.data):
            try:
                inp = self.tensor_to_object(pair.inp)
                tgt = self.tensor_to_object(pair.tgt)
                valid = validator(inp, tgt)
            except Exception:
                valid = False
            if valid:
                keep_raw.append(raw)
                keep_data.append(pair)
            else:
                removed += 1
        if removed:
            self.raw_data = keep_raw
            self.data = keep_data
            self.checksums = self._build_index()
            self._snapshot()
        return removed

    def shuffle(self, *, generator: torch.Generator | None = None) -> None:
        """Randomly shuffle dataset order in-place."""

        idx = torch.randperm(len(self.data), generator=generator).tolist()
        self.raw_data = [self.raw_data[i] for i in idx]
        self.data = [self.data[i] for i in idx]
        self.checksums = self._build_index()
        self._snapshot()

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

    def split_deterministic(
        self,
        train_ratio: float,
        val_ratio: float,
        *,
        salt: str = "",
    ) -> tuple["BitTensorDataset", "BitTensorDataset", "BitTensorDataset"]:
        """Deterministically split dataset into train/val/test using hashing."""

        if not 0.0 < train_ratio < 1.0:
            raise ValueError("train_ratio must be between 0 and 1")
        if not 0.0 <= val_ratio < 1.0:
            raise ValueError("val_ratio must be between 0 and 1")
        if train_ratio + val_ratio >= 1.0:
            raise ValueError("train_ratio + val_ratio must be < 1")

        scored: list[tuple[float, tuple[Any, Any]]] = []
        for inp, out in self.iter_decoded():
            h = hashlib.sha256(
                object_to_bytes(inp, serializer=self.serializer)
                + object_to_bytes(out, serializer=self.serializer)
                + salt.encode("utf-8")
            ).digest()
            frac = int.from_bytes(h[:8], "big") / 2**64
            scored.append((frac, (inp, out)))

        scored.sort(key=lambda x: x[0])

        train_cut = train_ratio
        val_cut = train_ratio + val_ratio

        train_raw: list[tuple[Any, Any]] = []
        val_raw: list[tuple[Any, Any]] = []
        test_raw: list[tuple[Any, Any]] = []

        for frac, pair in scored:
            if frac < train_cut:
                train_raw.append(pair)
            elif frac < val_cut:
                val_raw.append(pair)
            else:
                test_raw.append(pair)

        def _make(raw: list[tuple[Any, Any]]) -> "BitTensorDataset":
            return BitTensorDataset(
                raw,
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
            )

        return _make(train_raw), _make(val_raw), _make(test_raw)

    def merge(self, other: "BitTensorDataset", *, prefer: str = "self") -> "BitTensorDataset":
        """Merge two datasets resolving conflicts by ``prefer``."""

        if prefer not in {"self", "other", "raise"}:
            raise ValueError("prefer must be 'self', 'other' or 'raise'")

        result: dict[str, tuple[Any, Any]] = {
            hashlib.sha256(object_to_bytes(i, serializer=self.serializer)).hexdigest(): (i, t)
            for i, t in self.iter_decoded()
        }
        for inp, tgt in other.iter_decoded():
            key = hashlib.sha256(object_to_bytes(inp, serializer=self.serializer)).hexdigest()
            if key in result:
                existing = result[key][1]
                if existing != tgt:
                    if prefer == "other":
                        result[key] = (inp, tgt)
                    elif prefer == "raise":
                        raise ValueError("conflicting target for input")
            else:
                result[key] = (inp, tgt)

        merged_raw = list(result.values())
        return BitTensorDataset(
            merged_raw,
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
        )

    @classmethod
    def cached(
        cls,
        data: Iterable[tuple[Any, Any]],
        cache_path: str,
        **kwargs,
    ) -> "BitTensorDataset":
        """Load dataset from ``cache_path`` or create and cache it."""

        if os.path.exists(cache_path):
            return cls.load(cache_path)
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        ds = cls(list(data), **kwargs)
        ds.save(cache_path)
        return ds

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

    # --- Modification history helpers ---
    def _snapshot(self) -> None:
        self._history = self._history[: self._history_index + 1]
        self._history.append([(a, b) for a, b in self.raw_data])
        self._history_index += 1

    def _rebuild_from_raw(self) -> None:
        self.data = []
        for inp, out in self.raw_data:
            pair = self.pair_pool.allocate()
            pair.inp = self._obj_to_tensor(inp)
            pair.tgt = self._obj_to_tensor(out)
            self.data.append(pair)
        self.checksums = self._build_index()

    def undo(self, steps: int = 1) -> None:
        """Revert dataset state ``steps`` back in history."""

        for _ in range(steps):
            if self._history_index <= 0:
                break
            self._history_index -= 1
            self.raw_data = [(a, b) for a, b in self._history[self._history_index]]
        self._rebuild_from_raw()

    def redo(self, steps: int = 1) -> None:
        """Reapply undone modifications."""

        for _ in range(steps):
            if self._history_index + 1 >= len(self._history):
                break
            self._history_index += 1
            self.raw_data = [(a, b) for a, b in self._history[self._history_index]]
        self._rebuild_from_raw()

    # --- Sample level transforms and async save ---
    def transform_samples(
        self,
        input_transform: Callable[[Any], Any] | None = None,
        target_transform: Callable[[Any], Any] | None = None,
        *,
        rebuild_vocab: bool = False,
    ) -> None:
        """Apply per-sample transforms to all pairs."""

        def _fn(inp: Any, tgt: Any) -> tuple[Any, Any]:
            return (
                input_transform(inp) if input_transform else inp,
                target_transform(tgt) if target_transform else tgt,
            )

        self.map_pairs(_fn, rebuild_vocab=rebuild_vocab)

    def save_async(self, path: str) -> threading.Thread:
        """Persist dataset in a background thread."""

        def _task() -> None:
            self.save(path)

        t = threading.Thread(target=_task, daemon=True)
        t.start()
        return t

    # --- Approximate nearest neighbour search ---
    def build_ann_index(self, num_trees: int = 10) -> None:
        """Build an Annoy index for the dataset using Hamming distance."""
        try:
            from annoy import AnnoyIndex
        except Exception as e:  # pragma: no cover - import failure
            raise RuntimeError("annoy package required") from e

        dim = self[0][0].numel()
        self._ann_index = AnnoyIndex(dim, "hamming")
        for idx, (inp, _) in enumerate(self):
            self._ann_index.add_item(idx, inp.flatten().tolist())
        self._ann_index.build(num_trees)

    def nearest_neighbors(self, tensor: torch.Tensor, k: int = 5) -> list[int]:
        """Return indices of ``k`` nearest inputs to ``tensor``."""
        if hasattr(self, "_ann_index"):
            return self._ann_index.get_nns_by_vector(tensor.flatten().tolist(), k)
        # Fallback using torch.cdist
        inputs = torch.stack([p.inp.flatten() for p in self.data])
        dists = torch.cdist(inputs.float(), tensor.flatten().unsqueeze(0).float())
        _, idx = torch.topk(dists.squeeze(1), k, largest=False)
        return idx.tolist()
