import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from marble_to_pytorch import convert_core
from pytorch_to_marble import convert_model


class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 3),
            nn.ReLU(),
            nn.Linear(3, 2),
        )

    def forward(self, x):
        return self.net(x)


def test_round_trip_linear_model():
    torch.manual_seed(0)
    model = SimpleMLP()
    model.input_size = 4
    x = torch.randn(1, 4)
    with torch.no_grad():
        expected = model(x)
    core = convert_model(model)
    rebuilt = convert_core(core)
    with torch.no_grad():
        got = rebuilt(x)
    assert torch.allclose(got, expected, atol=1e-4)
