import asyncio
import csv
from pathlib import Path

import torch

from branch_container import BranchContainer
from dataset_loader import load_dataset


def load_ids(source, *, num_shards=None, shard_index=0, device="cpu"):
    pairs = load_dataset(source, num_shards=num_shards, shard_index=shard_index)
    ids = [int(a) for a, _ in pairs]
    return torch.tensor(ids, device=device)


def test_distribute_shards_across_branches(tmp_path):
    csv_path = Path(tmp_path) / "data.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["input", "target"])
        for i in range(4):
            writer.writerow([i, i])
    step = {
        "name": "load_ids",
        "module": __name__,
        "func": "load_ids",
        "params": {"source": str(csv_path)},
    }
    branches = [[step], [step]]
    container = BranchContainer(branches)
    results = asyncio.run(container.run(None))
    ids0 = set(results[0].tolist())
    ids1 = set(results[1].tolist())
    assert ids0.isdisjoint(ids1)
    assert ids0 | ids1 == {0, 1, 2, 3}
    if torch.cuda.is_available():
        assert results[0].device.type == results[1].device.type
