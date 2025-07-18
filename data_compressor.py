import numpy as np
import zlib

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
