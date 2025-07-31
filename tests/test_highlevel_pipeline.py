import os
import sys
import yaml
from tqdm import tqdm as std_tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import marble_imports
import marble_brain
import marble_main
from marble_base import MetricsVisualizer
from tests.test_core_functions import minimal_params

import marble_interface
from highlevel_pipeline import HighLevelPipeline


def _config_path(tmp_path):
    cfg = {"core": minimal_params(), "brain": {"save_dir": str(tmp_path)}}
    p = tmp_path / "cfg.yaml"
    with open(p, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f)
    return p


def test_highlevel_pipeline_runs(tmp_path):
    marble_imports.tqdm = std_tqdm
    marble_brain.tqdm = std_tqdm
    marble_main.MetricsVisualizer = MetricsVisualizer

    cfg = _config_path(tmp_path)
    hp = HighLevelPipeline()
    hp.new_marble_system(config_path=str(cfg))
    hp.train_marble_system(train_examples=[(0.1, 0.2)], epochs=1)
    marble, results = hp.execute()
    assert isinstance(marble, marble_interface.MARBLE)
    assert len(results) == 2


def test_highlevel_pipeline_save_load(tmp_path):
    hp = HighLevelPipeline()
    hp.add_step("new_marble_system", module="marble_interface", params={})
    json_path = tmp_path / "pipe.json"
    hp.save_json(json_path)
    with open(json_path, "r", encoding="utf-8") as f:
        loaded = HighLevelPipeline.load_json(f)
    assert loaded.steps == hp.steps


def test_highlevel_pipeline_cross_module(tmp_path):
    marble_imports.tqdm = std_tqdm
    marble_brain.tqdm = std_tqdm
    marble_main.MetricsVisualizer = MetricsVisualizer

    cfg = _config_path(tmp_path)
    hp = HighLevelPipeline()
    hp.plugin_system.load_plugins(dirs=[])
    hp.new_marble_system(config_path=str(cfg))
    marble, results = hp.execute()
    assert isinstance(marble, marble_interface.MARBLE)
    assert results[0] is None
