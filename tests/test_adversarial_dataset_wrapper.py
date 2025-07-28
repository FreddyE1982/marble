import torch
from torch.utils.data import TensorDataset
from torch import nn
from adversarial_dataset_wrapper import AdversarialDataset


def test_adversarial_dataset_wraps():
    base = TensorDataset(torch.zeros(2, 2), torch.tensor([0, 1]))
    model = nn.Linear(2, 2)
    wrapped = AdversarialDataset(base, model, epsilon=0.1)
    x, y = wrapped[0]
    assert x.shape == torch.Size([2])
    assert y.item() == 0
