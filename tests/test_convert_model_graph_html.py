import sys
from pathlib import Path
import subprocess

import torch


def _small_model() -> torch.nn.Module:
    # Use only built-in layers so loading via torch.load works in subprocess
    return torch.nn.Sequential(torch.nn.Linear(2, 1))


def test_convert_model_summary_graph(tmp_path):
    model = _small_model()
    model_path = tmp_path / "model.pt"
    torch.save(model, model_path)
    html_path = tmp_path / "summary.html"
    script = Path(__file__).resolve().parent.parent / "convert_model.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--pytorch",
            str(model_path),
            "--summary",
            "--summary-graph",
            str(html_path),
        ],
        capture_output=True,
    )
    assert result.returncode == 0
    assert html_path.exists()
    assert html_path.stat().st_size > 0
