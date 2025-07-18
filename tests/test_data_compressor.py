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
