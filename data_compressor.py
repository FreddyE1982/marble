import pickle
import struct
import zlib

import numpy as np


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
    """

    def __init__(
        self,
        level: int = 6,
        compression_enabled: bool = True,
        delta_encoding: bool = False,
        algorithm: str = "zlib",
    ) -> None:
        self.level = level
        self.compression_enabled = compression_enabled
        self.delta_encoding = delta_encoding
        self.algorithm = algorithm

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
        if self.algorithm == "zlib":
            return zlib.compress(bits.tobytes(), self.level)
        if self.algorithm == "lzma":
            import lzma

            return lzma.compress(bits.tobytes(), preset=self.level)
        raise ValueError(f"Unknown compression algorithm {self.algorithm}")

    def decompress(self, compressed: bytes) -> bytes:
        """Decompress and convert the binary representation back to bytes."""
        if not self.compression_enabled:
            return compressed
        if self.algorithm == "zlib":
            bits_bytes = zlib.decompress(compressed)
        elif self.algorithm == "lzma":
            import lzma

            bits_bytes = lzma.decompress(compressed)
        else:
            raise ValueError(f"Unknown compression algorithm {self.algorithm}")
        bits = np.frombuffer(bits_bytes, dtype=np.uint8)
        original_bytes = self.bits_to_bytes(bits)
        return original_bytes

    def compress_array(self, array: np.ndarray) -> bytes:
        """Compress a NumPy array with metadata for lossless recovery."""
        metadata = {"shape": array.shape, "dtype": str(array.dtype)}
        arr = array
        if self.delta_encoding:
            flat = arr.ravel()
            deltas = np.diff(flat, prepend=flat[0]).astype(arr.dtype)
            metadata["delta_encoding"] = True
            metadata["first_value"] = flat[0].item()
            arr_bytes = deltas.tobytes()
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
        arr = np.frombuffer(array_bytes, dtype=np.dtype(metadata["dtype"]))
        if metadata.get("delta_encoding"):
            first_val = np.asarray(metadata.get("first_value"), dtype=arr.dtype)
            arr = np.cumsum(arr)
            arr = arr + first_val
        return arr.reshape(metadata["shape"])
