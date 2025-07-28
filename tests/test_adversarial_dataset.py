import os
import sys
import torch
from torch.utils.data import Dataset

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from adversarial_dataset import FGSMDataset


class ToyDataset(Dataset):
    def __init__(self):
        self.data = [(0.1, 0.2), (0.2, 0.4)]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)


def test_fgsm_dataset_produces_different_inputs():
    dataset = ToyDataset()
    model = ToyModel()
    wrapped = FGSMDataset(dataset, model, epsilon=0.5)
    x0, y0 = dataset[0]
    adv_x0, adv_y0 = wrapped[0]
    assert adv_y0 == y0
    assert adv_x0 != x0
