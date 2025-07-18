import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
from marble import DataLoader


def test_marble_dataloader_roundtrip():
    dl = DataLoader()
    data = {"a": 1, "b": [1, 2, 3]}
    tensor = dl.encode(data)
    restored = dl.decode(tensor)
    assert restored == data


def test_marble_dataloader_array_roundtrip():
    dl = DataLoader()
    arr = np.arange(6, dtype=np.float32).reshape(2, 3)
    tensor = dl.encode_array(arr)
    restored = dl.decode_array(tensor)
    assert np.array_equal(restored, arr)
