import torch
from marble_core import _W1, _B1, _W2, _B2, Core

class MarbleTorchAdapter(torch.nn.Module):
    """PyTorch module mirroring Marble's message passing MLP."""

    def __init__(self, core: Core) -> None:
        super().__init__()
        self.core = core
        self.w1 = torch.nn.Parameter(torch.tensor(_W1, dtype=torch.float32))
        self.b1 = torch.nn.Parameter(torch.tensor(_B1, dtype=torch.float32))
        self.w2 = torch.nn.Parameter(torch.tensor(_W2, dtype=torch.float32))
        self.b2 = torch.nn.Parameter(torch.tensor(_B2, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        h = torch.tanh(x @ self.w1 + self.b1)
        return torch.tanh(h @ self.w2 + self.b2)


def core_to_torch(core: Core) -> MarbleTorchAdapter:
    """Return a ``MarbleTorchAdapter`` wrapping ``core`` for inference."""
    return MarbleTorchAdapter(core)
