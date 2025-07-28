import torch
from torch import nn
from adversarial_utils import fgsm_attack


def test_fgsm_attack_changes_input():
    model = nn.Linear(2, 2)
    inputs = torch.randn(1, 2)
    labels = torch.tensor([1])
    adv = fgsm_attack(model, inputs, labels, epsilon=0.1)
    assert not torch.allclose(inputs, adv)
