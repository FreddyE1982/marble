import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
import torch
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


def test_marble_dataloader_disable_compression():
    dl = DataLoader(compression_enabled=False)
    data = {"x": 42}
    tensor = dl.encode(data)
    restored = dl.decode(tensor)
    assert restored == data


def test_marble_dataloader_tensor_roundtrip():
    dl = DataLoader()
    t = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    encoded = dl.encode_tensor(t)
    decoded = dl.decode_tensor(encoded)
    assert torch.allclose(decoded, t)


def test_encode_array_accepts_tensor():
    dl = DataLoader()
    t = torch.arange(4, dtype=torch.float32)
    encoded = dl.encode_array(t)
    decoded = dl.decode_array(encoded)
    assert np.allclose(decoded, t.numpy())


def test_marble_dataloader_custom_dtype():
    dl = DataLoader(tensor_dtype="int16")
    data = {"x": 123}
    tensor = dl.encode(data)
    assert str(tensor.dtype) == "int16"
    restored = dl.decode(tensor)
    assert restored == data


def test_marble_dataloader_array_custom_dtype():
    dl = DataLoader(tensor_dtype="int16")
    arr = np.arange(4, dtype=np.float32)
    tensor = dl.encode_array(arr)
    assert str(tensor.dtype) == "int16"
    decoded = dl.decode_array(tensor)
    assert np.allclose(decoded, arr)


def test_marble_dataloader_various_types_roundtrip():
    dl = DataLoader()
    samples = [
        "text",
        np.ones((2, 2), dtype=np.uint8),
        b"\x00\x01\x02",
    ]
    for item in samples:
        restored = dl.decode(dl.encode(item))
        if isinstance(item, np.ndarray):
            assert np.array_equal(restored, item)
        else:
            assert restored == item


class _NeverEqual:
    def __eq__(self, other):
        return False


def test_round_trip_penalty():
    dl = DataLoader(enable_round_trip_check=True, round_trip_penalty=0.5)
    obj = _NeverEqual()
    assert dl.round_trip_penalty_for(obj) == 0.5
