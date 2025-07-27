import torch
import marble_core
from marble_core import _REP_SIZE


def test_simple_mlp_mixed_precision():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.randn(2, _REP_SIZE, device=device)
    out = marble_core._simple_mlp(x, mixed_precision=True)
    assert out.shape == (2, _REP_SIZE)
    assert torch.all(torch.isfinite(out))
