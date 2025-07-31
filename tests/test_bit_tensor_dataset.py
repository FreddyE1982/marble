import os
import sys

import pytest
import torch

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


def test_vocab_options_max_size_and_occurrence():
    data = [("foo", "bar"), ("foo", "bar")]
    ds = BitTensorDataset(
        data,
        use_vocab=True,
        max_vocab_size=1,
        min_occurrence=1000,
    )
    assert ds.get_vocab() == {}


def test_vocab_only_mode_changes_output():
    data = [("a", "b"), ("c", "d"), ("e", "f")]
    ds = BitTensorDataset(data, use_vocab=True, mixed=False, max_vocab_size=1)
    t_in, _ = ds[0]
    with pytest.raises(Exception):
        ds.tensor_to_object(t_in)


def test_custom_vocab_reuse():
    data = [("x", "y")]
    ds1 = BitTensorDataset(data, use_vocab=True)
    vocab = ds1.get_vocab()
    ds2 = BitTensorDataset(data, vocab=vocab)
    assert ds2.get_vocab() == vocab
    obj = ds2.tensor_to_object(ds2[0][0])
    assert obj == "x"


def test_max_word_length_respected():
    data = [("abcd", "efgh"), ("ijkl", "mnop")]
    ds = BitTensorDataset(data, use_vocab=True, max_word_length=3)
    vocab = ds.get_vocab()
    assert all(len(pattern) <= 3 for pattern in vocab)


def test_dataset_device_setting():
    data = [(1, 2)]
    ds = BitTensorDataset(data, device="cpu")
    assert ds[0][0].device == torch.device("cpu")
    obj = ds.tensor_to_object(ds[0][0])
    assert obj == 1
