import os, sys, subprocess, json
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from tests.test_core_functions import minimal_params


def test_cli_pipeline(tmp_path):
    cfg = tmp_path / "cfg.yaml"
    import yaml

    cfg.write_text(yaml.safe_dump({"core": minimal_params()}))
    pipe_path = tmp_path / "pipe.json"
    pipeline = [{"func": "count_marble_synapses"}]
    pipe_path.write_text(json.dumps(pipeline))
    result = subprocess.run([
        sys.executable,
        "cli.py",
        "--config",
        str(cfg),
        "--pipeline",
        str(pipe_path),
    ], capture_output=True)
    assert result.returncode == 0
    assert b"[" in result.stdout

