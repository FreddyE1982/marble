import pickle
import struct
import zlib
from typing import Callable, Dict

import numpy as np
import torch

from quantized_tensor import QuantizedTensor
from sparse_utils import to_sparse, to_dense

ALGORITHMS: Dict[str, tuple[Callable[[bytes, int], bytes], Callable[[bytes], bytes]]] = {}


def register_algorithm(
    name: str,
    compress_fn: Callable[[bytes, int], bytes],
    decompress_fn: Callable[[bytes], bytes],
) -> None:
    """Register a custom compression algorithm."""

    ALGORITHMS[name] = (compress_fn, decompress_fn)


# register built-in algorithms
register_algorithm("zlib", lambda b, level: zlib.compress(b, level), zlib.decompress)
register_algorithm(
    "lzma",
    lambda b, level: __import__("lzma").compress(b, preset=level),
    lambda b: __import__("lzma").decompress(b),
)


class DataCompressor:
    """Full transparent transitive binary compressor.

    Parameters
    ----------
    level : int, optional
        Compression level passed to :func:`zlib.compress`. ``0`` disables
        compression while ``9`` yields maximum compression. Defaults to ``6``.
    compression_enabled : bool, optional
        When ``False`` no compression or decompression is performed and data is
        returned unchanged.
    delta_encoding : bool, optional
        When ``True`` arrays are delta encoded prior to compression which can
        substantially improve ratios on smoothly varying data. Defaults to
        ``False``.
        algorithm : str, optional
        Compression algorithm to use. ``"zlib"`` provides fast general
        compression while ``"lzma"`` offers higher ratios at the cost of speed.
        Defaults to ``"zlib"``.
    quantization_bits : int, optional
        When greater than ``0`` arrays are quantized to the given bit width
        before compression using :class:`QuantizedTensor`. ``0`` disables
        quantization.
    sparse_threshold : float, optional
        If set, arrays with a fraction of non-zero elements below this value are
        converted to a CSR sparse matrix prior to compression.
    """

    def __init__(
        self,
        level: int = 6,
        compression_enabled: bool = True,
        delta_encoding: bool = False,
        algorithm: str = "zlib",
        quantization_bits: int = 0,
        sparse_threshold: float | None = None,
    ) -> None:
        self.level = level
        self.compression_enabled = compression_enabled
        self.delta_encoding = delta_encoding
        if algorithm not in ALGORITHMS:
            raise ValueError(f"Unknown compression algorithm {algorithm}")
        self.algorithm = algorithm
        self.quantization_bits = quantization_bits
        self.sparse_threshold = sparse_threshold

    @staticmethod
    def bytes_to_bits(data: bytes) -> np.ndarray:
        """Convert bytes to an array of binary bits."""
        byte_array = np.frombuffer(data, dtype=np.uint8)
        bits = np.unpackbits(byte_array)
        return bits

    @staticmethod
    def bits_to_bytes(bits: np.ndarray) -> bytes:
        """Convert an array of binary bits back to bytes."""
        byte_array = np.packbits(bits.astype(np.uint8))
        return byte_array.tobytes()

    def compress(self, data: bytes) -> bytes:
        """Compress bytes after converting to a binary representation."""
        if not self.compression_enabled:
            return data
        bits = self.bytes_to_bits(data)
        if self.algorithm in ALGORITHMS:
            fn, _ = ALGORITHMS[self.algorithm]
            return fn(bits.tobytes(), self.level)
        raise ValueError(f"Unknown compression algorithm {self.algorithm}")

    def decompress(self, compressed: bytes) -> bytes:
        """Decompress and convert the binary representation back to bytes."""
        if not self.compression_enabled:
            return compressed
        if self.algorithm in ALGORITHMS:
            _, fn = ALGORITHMS[self.algorithm]
            bits_bytes = fn(compressed)
        else:
            raise ValueError(f"Unknown compression algorithm {self.algorithm}")
        bits = np.frombuffer(bits_bytes, dtype=np.uint8)
        original_bytes = self.bits_to_bytes(bits)
        return original_bytes

    def compress_array(self, array: np.ndarray) -> bytes:
        """Compress a NumPy array with metadata for lossless recovery."""
        metadata = {"shape": array.shape, "dtype": str(array.dtype)}

        # sparse conversion happens prior to any other transform
        if (
            self.sparse_threshold is not None
            and array.size > 0
            and (np.count_nonzero(array) / array.size) < self.sparse_threshold
        ):
            sparse_arr = to_sparse(array)
            metadata["sparse"] = True
            arr_bytes = pickle.dumps(sparse_arr)
        else:
            arr = array
            if self.delta_encoding:
                flat = arr.ravel()
                deltas = np.diff(flat, prepend=flat[0]).astype(arr.dtype)
                metadata["delta_encoding"] = True
                metadata["first_value"] = flat[0].item()
                arr = deltas
            if self.quantization_bits > 0:
                tensor = torch.from_numpy(arr.astype(np.float32))
                q = QuantizedTensor.from_tensor(
                    tensor, bit_width=self.quantization_bits
                )
                metadata.update(
                    {
                        "quantized": True,
                        "scale": q.scale,
                        "zero_point": q.zero_point,
                        "bit_width": q.bit_width,
                    }
                )
                arr_bytes = q.bits.cpu().numpy().tobytes()
            else:
                arr_bytes = arr.tobytes()

        meta_bytes = pickle.dumps(metadata)
        header = struct.pack("<I", len(meta_bytes))
        payload = header + meta_bytes + arr_bytes
        return self.compress(payload)

    def decompress_array(self, compressed: bytes) -> np.ndarray:
        """Decompress bytes into a NumPy array using stored metadata."""
        payload = self.decompress(compressed)
        meta_len = struct.unpack("<I", payload[:4])[0]
        meta_bytes = payload[4 : 4 + meta_len]
        metadata = pickle.loads(meta_bytes)
        array_bytes = payload[4 + meta_len :]

        if metadata.get("sparse"):
            sparse_arr = pickle.loads(array_bytes)
            arr = to_dense(sparse_arr)
        elif metadata.get("quantized"):
            bits = torch.frombuffer(array_bytes, dtype=torch.uint8)
            qt = QuantizedTensor(
                bits=bits,
                shape=tuple(metadata["shape"]),
                scale=float(metadata["scale"]),
                zero_point=int(metadata["zero_point"]),
                bit_width=int(metadata["bit_width"]),
                device=torch.device("cpu"),
            )
            arr = qt.to_dense().cpu().numpy().astype(metadata["dtype"])
        else:
            arr = np.frombuffer(array_bytes, dtype=np.dtype(metadata["dtype"]))

        if metadata.get("delta_encoding") and not metadata.get("sparse"):
            first_val = np.asarray(metadata.get("first_value"), dtype=arr.dtype)
            arr = np.cumsum(arr)
            arr = arr + first_val

        return arr.reshape(metadata["shape"])
