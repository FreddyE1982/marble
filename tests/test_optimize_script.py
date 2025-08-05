"""Tests for Optuna optimisation script."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import yaml


def test_optimize_script_runs_and_creates_files(tmp_path: Path) -> None:
    db_path = tmp_path / "optuna_db.sqlite3"
    out_path = tmp_path / "best_params.yaml"

    cmd = [
        sys.executable,
        "scripts/optimize.py",
        "--trials",
        "1",
        "--storage",
        f"sqlite:///{db_path}",
        "--output",
        str(out_path),
    ]
    subprocess.run(cmd, check=True, cwd=Path(__file__).resolve().parents[1])

    assert db_path.exists(), "Optuna database was not created"
    assert out_path.exists(), "Best params file was not created"
    data = yaml.safe_load(out_path.read_text())
    assert isinstance(data, dict) and data, "Best params YAML is empty"
