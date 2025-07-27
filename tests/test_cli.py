import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from tests.test_core_functions import minimal_params


def test_cli_help():
    result = subprocess.run([sys.executable, "cli.py", "--help"], capture_output=True)
    assert result.returncode == 0
    assert b"MARBLE command line interface" in result.stdout


def test_cli_no_train(tmp_path):
    cfg = Path(tmp_path) / "cfg.yaml"
    import yaml

    cfg.write_text(yaml.safe_dump({"core": minimal_params()}))
    result = subprocess.run(
        [
            sys.executable,
            "cli.py",
            "--config",
            str(cfg),
        ]
    )
    assert result.returncode == 0


def test_cli_export_core(tmp_path):
    cfg = Path(tmp_path) / "cfg.yaml"
    import yaml

    cfg.write_text(yaml.safe_dump({"core": minimal_params()}))
    export_path = tmp_path / "core.json"
    result = subprocess.run(
        [
            sys.executable,
            "cli.py",
            "--config",
            str(cfg),
            "--export-core",
            str(export_path),
        ]
    )
    assert result.returncode == 0
    assert export_path.exists()


def test_cli_scheduler_override(tmp_path):
    cfg = Path(tmp_path) / "cfg.yaml"
    import yaml

    cfg.write_text(yaml.safe_dump({"core": minimal_params()}))
    result = subprocess.run(
        [
            sys.executable,
            "cli.py",
            "--config",
            str(cfg),
            "--lr-scheduler",
            "exponential",
            "--scheduler-gamma",
            "0.9",
        ]
    )
    assert result.returncode == 0


def test_cli_grid_search(tmp_path):
    cfg = Path(tmp_path) / "cfg.yaml"
    import yaml

    cfg.write_text(yaml.safe_dump({"core": minimal_params()}))
    grid_file = tmp_path / "grid.yaml"
    grid_file.write_text(yaml.safe_dump({"dropout_probability": [0.0, 0.1]}))
    result = subprocess.run(
        [
            sys.executable,
            "cli.py",
            "--config",
            str(cfg),
            "--grid-search",
            str(grid_file),
        ],
        capture_output=True,
    )
    assert result.returncode == 0
    assert b"dropout_probability" in result.stdout
