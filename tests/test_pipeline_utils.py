import os
import sys
from pipeline import Pipeline
from config_loader import create_marble_from_config
from tests.test_core_functions import minimal_params
import yaml


def test_diff_config():
    p1 = Pipeline([{"func": "a"}])
    p2 = Pipeline([{"func": "b"}])
    diff = p1.diff_config(p2.steps)
    assert '"func": "a"' in diff and '"func": "b"' in diff


def test_log_callback(tmp_path):
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(yaml.safe_dump({"core": minimal_params()}))
    marble = create_marble_from_config(str(cfg))
    pipe = Pipeline([{"func": "count_marble_synapses"}])
    logs = []
    pipe.execute(marble, log_callback=logs.append)
    assert logs
