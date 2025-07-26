import os
import sys
import subprocess
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
    result = subprocess.run([
        sys.executable,
        "cli.py",
        "--config",
        str(cfg),
    ])
    assert result.returncode == 0


def test_cli_export_core(tmp_path):
    cfg = Path(tmp_path) / "cfg.yaml"
    import yaml

    cfg.write_text(yaml.safe_dump({"core": minimal_params()}))
    export_path = tmp_path / "core.json"
    result = subprocess.run([
        sys.executable,
        "cli.py",
        "--config",
        str(cfg),
        "--export-core",
        str(export_path),
    ])
    assert result.returncode == 0
    assert export_path.exists()
