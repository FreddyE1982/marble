import os
import sys
import subprocess
from pathlib import Path

import torch


def _small_model() -> torch.nn.Module:
    return torch.nn.Sequential(torch.nn.Linear(2, 1))


def test_convert_model_show_graph(tmp_path):
    model = _small_model()
    model_path = tmp_path / "model.pt"
    torch.save(model, model_path)
    script = Path(__file__).resolve().parent.parent / "convert_model.py"
    env = os.environ.copy()
    env["BROWSER"] = "echo"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--pytorch",
            str(model_path),
            "--show-graph",
            "--dry-run",
        ],
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 0
    assert "Graph HTML saved to" in result.stdout
