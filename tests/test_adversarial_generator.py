import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from adversarial_generator import fgsm_generate


class ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)


def test_fgsm_generate_returns_examples():
    model = ToyModel()
    xs = [torch.tensor([0.5])] * 2
    ys = [torch.tensor([1.0])] * 2
    adv = fgsm_generate(model, xs, ys, epsilon=0.1)
    assert len(adv) == 2
    assert not torch.equal(adv[0][0], xs[0])
