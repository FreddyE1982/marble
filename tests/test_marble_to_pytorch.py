import torch
from torch import nn

from pytorch_to_marble import convert_model
from marble_to_pytorch import convert_core
from tests.test_core_functions import minimal_params


class SmallNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(2, 3),
            nn.ReLU(),
            nn.Linear(3, 1),
        )
        self.input_size = 2

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":  # pragma: no cover - trivial
        import torch

        return self.seq(x)


def _train_model(model: nn.Module) -> None:
    """Train ``model`` on a simple regression task for a few steps."""
    import torch

    x = torch.randn(20, 2)
    y = torch.randn(20, 1)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    for _ in range(50):
        opt.zero_grad()
        loss = ((model(x) - y) ** 2).mean()
        loss.backward()
        opt.step()


def test_round_trip_conversion():
    import torch

    model = SmallNet()
    _train_model(model)
    core = convert_model(model, core_params=minimal_params())
    rebuilt = convert_core(core)

    with torch.no_grad():
        for p1, p2 in zip(model.parameters(), rebuilt.parameters()):
            assert torch.allclose(p1, p2, atol=1e-5)
    inp = torch.randn(5, 2)
    out1 = model(inp)
    out2 = rebuilt(inp)
    assert torch.allclose(out1, out2, atol=1e-5)
