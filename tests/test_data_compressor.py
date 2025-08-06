import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np

from data_compressor import DataCompressor
from data_compressor import register_algorithm


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


def test_delta_encoding_roundtrip():
    dc = DataCompressor(delta_encoding=True)
    arr = np.arange(10, dtype=np.int32)
    compressed = dc.compress_array(arr)
    restored = dc.decompress_array(compressed)
    assert np.array_equal(restored, arr)


def test_lzma_algorithm():
    dc = DataCompressor(algorithm="lzma")
    data = b"compress me" * 10
    out = dc.decompress(dc.compress(data))
    assert out == data


def test_plugin_algorithm():
    register_algorithm("reverse", lambda b, lvl: b[::-1], lambda b: b[::-1])
    dc = DataCompressor(algorithm="reverse")
    data = b"plugin-data"
    assert dc.decompress(dc.compress(data)) == data


def test_quantized_array_roundtrip():
    dc = DataCompressor(quantization_bits=4)
    arr = np.linspace(-1, 1, 16, dtype=np.float32)
    comp = dc.compress_array(arr)
    restored = dc.decompress_array(comp)
    assert np.allclose(restored, arr, atol=0.1)


def test_sparse_array_roundtrip():
    arr = np.zeros((10, 10), dtype=np.float32)
    arr[0, 1] = 1.0
    dc = DataCompressor(sparse_threshold=0.2)
    comp = dc.compress_array(arr)
    restored = dc.decompress_array(comp)
    assert np.array_equal(restored, arr)
