"""Dataset wrapper that returns adversarial examples alongside originals."""

from __future__ import annotations

import torch
from torch.utils.data import Dataset

from adversarial_utils import fgsm_attack


class AdversarialDataset(Dataset):
    """Wrap a base dataset to produce adversarial samples."""

    def __init__(self, base: Dataset, model: torch.nn.Module, epsilon: float = 0.01):
        self.base = base
        self.model = model
        self.epsilon = float(epsilon)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.base)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:  # type: ignore[override]
        x, y = self.base[idx]
        x_adv = fgsm_attack(self.model, x.unsqueeze(0), y.unsqueeze(0), self.epsilon)[0]
        return x_adv, y
