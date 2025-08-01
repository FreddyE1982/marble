import os
import sys
import subprocess
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from tests.test_core_functions import minimal_params
import yaml
from pipeline import Pipeline


def test_pipeline_cli(tmp_path):
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(yaml.safe_dump({"core": minimal_params()}))
    pipe_path = tmp_path / "pipe.json"
    pipe = Pipeline()
    pipe.add_step("count_marble_synapses")
    pipe.save_json(pipe_path)

    result = subprocess.run(
        [sys.executable, "pipeline_cli.py", str(pipe_path), "--config", str(cfg)],
        capture_output=True,
    )
    assert result.returncode == 0
