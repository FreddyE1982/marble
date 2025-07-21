import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import yaml
from unittest import mock

import pandas as pd

from tests.test_core_functions import minimal_params

from streamlit_playground import load_examples, initialize_marble


def test_load_examples(tmp_path):
    path = tmp_path / "data.csv"
    path.write_text("input,target\n0.1,0.2\n0.2,0.4\n")
    with open(path, "r", encoding="utf-8") as f:
        ex = load_examples(f)
    assert ex == [(0.1, 0.2), (0.2, 0.4)]


def test_initialize_marble(tmp_path):
    cfg = {"core": minimal_params(), "brain": {"save_dir": str(tmp_path)}}
    cfg_path = tmp_path / "cfg.yaml"
    with open(cfg_path, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f)
    m = initialize_marble(str(cfg_path))
    assert len(m.get_core().neurons) > 0
