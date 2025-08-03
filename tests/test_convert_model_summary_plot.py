import subprocess
import sys
from pathlib import Path

import torch


class SmallModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = torch.nn.Linear(2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - simple pass-through
        return self.fc(x)


def test_convert_model_summary_plot(tmp_path):
    model = SmallModel()
    model_path = tmp_path / "model.pt"
    torch.save(model, model_path)
    plot_path = tmp_path / "summary.png"
    script = Path(__file__).resolve().parent.parent / "convert_model.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--pytorch",
            str(model_path),
            "--summary-plot",
            str(plot_path),
        ],
        capture_output=True,
    )
    assert result.returncode == 0
    assert plot_path.exists()
    assert plot_path.stat().st_size > 0
