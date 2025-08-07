import pytest
import torch
from marble_neuronenblitz import Neuronenblitz
from marble_core import Core
from bit_tensor_dataset import BitTensorDataset


def make_shard(start: int) -> BitTensorDataset:
    pairs = [
        (torch.tensor([i], dtype=torch.float32), torch.tensor([i + 1], dtype=torch.float32))
        for i in range(start, start + 4)
    ]
    return BitTensorDataset(pairs, device="cpu")

def test_train_streaming_shards_cpu():
    shards = [make_shard(0), make_shard(4)]
    nb = Neuronenblitz(Core({}))
    nb.train_streaming_shards(shards, batch_size=2, device="cpu")
    assert len(nb.training_history) > 0

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_train_streaming_shards_gpu():
    shards = [make_shard(0)]
    nb = Neuronenblitz(Core({}))
    nb.train_streaming_shards(shards, batch_size=2, device="cuda")
    assert len(nb.training_history) > 0
