import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
from marble_core import DataLoader
from PIL import Image
import pytest


def make_data(typ):
    if typ == "text":
        return "hello"
    if typ == "image":
        return Image.new("RGB", (2, 2), color=(123, 222, 111))
    if typ == "audio":
        return np.arange(8, dtype=np.float32) / 8.0
    if typ == "blob":
        return b"\x00\x01\x02\x03"
    raise ValueError(f"unknown type: {typ}")


@pytest.mark.parametrize(
    "input_type,target_type",
    [
        ("text", "image"),
        ("image", "text"),
        ("text", "audio"),
        ("audio", "text"),
        ("image", "audio"),
        ("audio", "image"),
        ("blob", "text"),
        ("text", "blob"),
        ("image", "blob"),
        ("blob", "image"),
        ("blob", "audio"),
        ("audio", "blob"),
    ],
)
def test_dataloader_multimodal_pairs_roundtrip(input_type, target_type):
    dl = DataLoader()
    sample = {"input": make_data(input_type), "target": make_data(target_type)}
    encoded = dl.encode(sample)
    decoded = dl.decode(encoded)
    assert type(decoded["input"]) == type(sample["input"])
    assert type(decoded["target"]) == type(sample["target"])
    if isinstance(sample["input"], np.ndarray):
        assert np.array_equal(decoded["input"], sample["input"])
    elif isinstance(sample["input"], Image.Image):
        assert decoded["input"].size == sample["input"].size
    else:
        assert decoded["input"] == sample["input"]
    if isinstance(sample["target"], np.ndarray):
        assert np.array_equal(decoded["target"], sample["target"])
    elif isinstance(sample["target"], Image.Image):
        assert decoded["target"].size == sample["target"].size
    else:
        assert decoded["target"] == sample["target"]
