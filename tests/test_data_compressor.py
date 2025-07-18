import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data_compressor import DataCompressor


def test_compression_roundtrip():
    dc = DataCompressor()
    original = b"example data bytes"
    compressed = dc.compress(original)
    decompressed = dc.decompress(compressed)
    assert decompressed == original
