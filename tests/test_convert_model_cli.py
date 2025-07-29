import sys
import subprocess

from pathlib import Path

import torch
from marble_interface import load_marble_system


class SmallModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = torch.nn.Linear(2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


def test_convert_model_marble(tmp_path):
    model = SmallModel()
    model_path = tmp_path / "model.pt"
    torch.save(model, model_path)

    out_path = tmp_path / "model.marble"
    script = Path(__file__).resolve().parent.parent / "convert_model.py"
    result = subprocess.run(
        [sys.executable, str(script), "--pytorch", str(model_path), "--output", str(out_path)],
        capture_output=True,
    )
    assert result.returncode == 0
    assert out_path.exists()

    marble = load_marble_system(str(out_path))
    assert len(marble.get_core().neurons) >= 2
