from __future__ import annotations

from typing import List, Tuple
import torch


class BitTensorDatasetHarness:
    """Generate simple bit-tensor datasets for testing."""

    def __init__(self, num_samples: int = 4) -> None:
        self.num_samples = num_samples

    def make_pairs(self) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        pairs = []
        for i in range(self.num_samples):
            inp = torch.randint(0, 2, (8, 8), dtype=torch.uint8)
            tgt = torch.randint(0, 2, (8, 8), dtype=torch.uint8)
            pairs.append((inp, tgt))
        return pairs
