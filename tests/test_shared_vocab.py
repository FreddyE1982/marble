import torch

from bit_tensor_dataset import (
    BitTensorDataset,
    bytes_to_tensors,
    tensors_to_bytes,
)
from shared_vocab import build_shared_vocab


def test_build_shared_vocab():
    data1 = [("a", "b")]
    data2 = [("c", "d")]
    vocab = build_shared_vocab([data1, data2], min_len=2, max_len=2, max_size=1, start_id=500, min_occurrence=1)
    ds1 = BitTensorDataset(data1, vocab=vocab)
    ds2 = BitTensorDataset(data2, vocab=vocab)
    assert ds1.get_vocab() == vocab
    assert ds2.get_vocab() == vocab
    assert ds1.tensor_to_object(ds1[0][0]) == "a"
    assert ds2.tensor_to_object(ds2[0][0]) == "c"


def test_bytes_tensor_gpu_cpu_parity():
    b = b"hello"
    cpu = bytes_to_tensors(b)
    assert tensors_to_bytes(cpu) == b
    if torch.cuda.is_available():
        gpu = bytes_to_tensors(b, device="cuda")
        assert torch.equal(gpu.cpu(), cpu)
        assert tensors_to_bytes(gpu.cpu()) == b

