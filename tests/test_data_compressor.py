import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np

from data_compressor import DataCompressor


def test_compression_roundtrip():
    dc = DataCompressor()
    original = b"example data bytes"
    compressed = dc.compress(original)
    decompressed = dc.decompress(compressed)
    assert decompressed == original


def test_array_compression_roundtrip():
    dc = DataCompressor()
    arr = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    compressed = dc.compress_array(arr)
    restored = dc.decompress_array(compressed)
    assert np.array_equal(restored, arr)


def test_custom_compression_level():
    low = DataCompressor(level=1)
    high = DataCompressor(level=9)
    data = b"data" * 100
    size_low = len(low.compress(data))
    size_high = len(high.compress(data))
    assert size_high <= size_low


def test_compression_toggle():
    dc = DataCompressor(compression_enabled=False)
    data = b"abc" * 10
    compressed = dc.compress(data)
    assert compressed == data
    assert dc.decompress(compressed) == data
