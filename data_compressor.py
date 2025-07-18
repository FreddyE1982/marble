import numpy as np
import zlib
import struct
import pickle

class DataCompressor:
    """Full transparent transitive binary compressor."""

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
        bits = self.bytes_to_bits(data)
        compressed = zlib.compress(bits.tobytes())
        return compressed

    def decompress(self, compressed: bytes) -> bytes:
        """Decompress and convert the binary representation back to bytes."""
        bits_bytes = zlib.decompress(compressed)
        bits = np.frombuffer(bits_bytes, dtype=np.uint8)
        original_bytes = self.bits_to_bytes(bits)
        return original_bytes

    def compress_array(self, array: np.ndarray) -> bytes:
        """Compress a NumPy array with metadata for lossless recovery."""
        metadata = {"shape": array.shape, "dtype": str(array.dtype)}
        meta_bytes = pickle.dumps(metadata)
        header = struct.pack("<I", len(meta_bytes))
        payload = header + meta_bytes + array.tobytes()
        return self.compress(payload)

    def decompress_array(self, compressed: bytes) -> np.ndarray:
        """Decompress bytes into a NumPy array using stored metadata."""
        payload = self.decompress(compressed)
        meta_len = struct.unpack("<I", payload[:4])[0]
        meta_bytes = payload[4 : 4 + meta_len]
        metadata = pickle.loads(meta_bytes)
        array_bytes = payload[4 + meta_len :]
        arr = np.frombuffer(array_bytes, dtype=np.dtype(metadata["dtype"]))
        return arr.reshape(metadata["shape"])
