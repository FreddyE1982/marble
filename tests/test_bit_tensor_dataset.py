import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from bit_tensor_dataset import BitTensorDataset


def test_roundtrip_no_vocab():
    data = [(123, {"a": 1}), (456, [1, 2, 3])]
    ds = BitTensorDataset(data)
    assert len(ds) == 2
    t_in, t_out = ds[0]
    obj_in = ds.tensor_to_object(t_in)
    obj_out = ds.tensor_to_object(t_out)
    assert obj_in == data[0][0]
    assert obj_out == data[0][1]


def test_roundtrip_with_vocab():
    data = [("hello", "world"), ("foo", "bar")]
    ds = BitTensorDataset(data, use_vocab=True)
    t_in, t_out = ds[1]
    obj_in = ds.tensor_to_object(t_in)
    obj_out = ds.tensor_to_object(t_out)
    assert obj_in == data[1][0]
    assert obj_out == data[1][1]
    assert ds.get_vocab() is not None

