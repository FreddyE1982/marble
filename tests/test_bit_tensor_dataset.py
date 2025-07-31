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


def test_bit_tensor_dataset_compression():
    data = [("long string" * 10, 123)]
    ds = BitTensorDataset(data, compress=True)
    encoded, _ = ds[0]
    decoded = ds.tensor_to_object(encoded)
    assert decoded == data[0][0]


def test_bit_tensor_dataset_iteration_and_save_load(tmp_path):
    pairs = [(1, 2), (3, 4)]
    ds = BitTensorDataset(pairs, start_id=500)
    assert list(ds) == [ds[0], ds[1]]
    save_path = tmp_path / "ds.pt"
    ds.save(save_path)
    loaded = BitTensorDataset.load(save_path)
    assert len(loaded) == 2
    assert loaded.tensor_to_object(loaded[0][0]) == 1
    assert loaded.start_id == 500


def test_bit_tensor_dataset_summary():
    pairs = [(1, 2), (3, 4)]
    ds = BitTensorDataset(pairs, use_vocab=True)
    info = ds.summary()
    assert info["num_pairs"] == 2
    assert info["vocab_size"] == ds.vocab_size()
    assert info["device"] == str(ds.device)
    assert info["compressed"] is False
    assert info["start_id"] == ds.start_id


def test_bit_tensor_dataset_add_extend():
    ds = BitTensorDataset([(0, 1)])
    ds.add_pair(2, 3)
    assert len(ds) == 2
    ds.extend([(4, 5), (6, 7)])
    assert len(ds) == 4
    assert ds.tensor_to_object(ds[2][0]) == 4


def test_bit_tensor_dataset_custom_start_id():
    data = [("aa", "bb"), ("cc", "dd")]
    ds = BitTensorDataset(data, use_vocab=True, start_id=700)
    vocab_vals = list(ds.get_vocab().values())
    assert vocab_vals and min(vocab_vals) >= 700
    assert ds.start_id == 700


def test_bit_tensor_dataset_iter_decoded_and_summary():
    data = [(1, 2), (3, 4)]
    ds = BitTensorDataset(data)
    decoded = list(ds.iter_decoded())
    assert decoded == data
    info = ds.summary()
    total = sum(a.numel() + b.numel() for a, b in ds)
    expected_avg = float(total) / len(ds)
    assert info["avg_pair_length"] == expected_avg
