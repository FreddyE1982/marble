import os
import sys

import torch
import yaml
from tqdm import tqdm as std_tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import marble_brain
import marble_imports
import marble_interface
import marble_main
from bit_tensor_dataset import BitTensorDataset
from highlevel_pipeline import HighLevelPipeline
from marble_base import MetricsVisualizer
from tests.test_core_functions import minimal_params


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


def test_highlevel_pipeline_detect_marble_in_nested(tmp_path):
    marble_imports.tqdm = std_tqdm
    marble_brain.tqdm = std_tqdm
    marble_main.MetricsVisualizer = MetricsVisualizer

    cfg = _config_path(tmp_path)

    def make_marble(marble=None):
        return {"hooked": True}, marble_interface.new_marble_system(
            config_path=str(cfg)
        )

    hp = HighLevelPipeline()
    hp.add_step(make_marble)
    hp.train_marble_system(train_examples=[(0.0, 0.0)], epochs=1)
    marble, results = hp.execute()
    assert isinstance(marble, marble_interface.MARBLE)
    assert isinstance(results[0], tuple)
    assert results[0][0] == {"hooked": True}


def test_highlevel_pipeline_nested_module(tmp_path):
    marble_imports.tqdm = std_tqdm
    marble_brain.tqdm = std_tqdm
    marble_main.MetricsVisualizer = MetricsVisualizer

    cfg = _config_path(tmp_path)
    hp = HighLevelPipeline()
    hp.new_marble_system(config_path=str(cfg))
    hp.marble_neuronenblitz.learning.disable_rl(nb=None)
    assert hp.steps[-1]["module"] == "marble_neuronenblitz.learning"


def test_highlevel_pipeline_default_bit_params():
    hp = HighLevelPipeline()
    ds = hp._maybe_bit_dataset([1, 2])
    assert isinstance(ds, BitTensorDataset)
    assert ds.mixed is True
    assert ds.max_vocab_size is None
    assert ds.min_word_length == 4
    assert ds.max_word_length == 8
    assert ds.min_occurrence == 4
    expected_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert ds.device == expected_device
    assert ds.compress is False
    assert ds.start_id == 256


def test_highlevel_pipeline_register_data_args():
    hp = HighLevelPipeline()

    def grab(custom=None):
        return custom

    hp.register_data_args("custom")
    hp.add_step(grab, params={"custom": [10, 11]})
    _, results = hp.execute()
    assert isinstance(results[0], BitTensorDataset)


def test_highlevel_pipeline_multi_step_chain(tmp_path):
    marble_imports.tqdm = std_tqdm
    marble_brain.tqdm = std_tqdm
    marble_main.MetricsVisualizer = MetricsVisualizer

    cfg = _config_path(tmp_path)
    hp = HighLevelPipeline()
    hp.plugin_system.load_plugins(dirs=[])
    hp.new_marble_system(config_path=str(cfg))
    hp.disable_marble_rl()
    hp.train_marble_system(train_examples=[(0, 0)], epochs=1)
    hp.enable_marble_rl()
    marble, results = hp.execute()
    assert isinstance(marble, marble_interface.MARBLE)
    assert len(results) == 5


def test_highlevel_pipeline_step_manipulation():
    hp = HighLevelPipeline()

    def a():
        return "a"

    def b():
        return "b"

    def c():
        return "c"

    hp.add_step(a)
    hp.add_step(b)
    hp.add_step(c)
    hp.move_step(2, 0)
    assert hp.steps[0]["callable"] == c
    hp.remove_step(1)
    assert len(hp.steps) == 2


def test_highlevel_pipeline_shared_vocab_param():
    vocab = {(1, 1): 300}
    hp = HighLevelPipeline(bit_dataset_params={"vocab": vocab})
    ds = hp._maybe_bit_dataset([0, 1])
    assert ds.get_vocab() == vocab


def test_highlevel_pipeline_custom_device():
    hp = HighLevelPipeline(bit_dataset_params={"device": "cpu"})
    ds = hp._maybe_bit_dataset([5, 6])
    assert ds.device == torch.device("cpu")


def test_highlevel_pipeline_compress_param():
    hp = HighLevelPipeline(bit_dataset_params={"compress": True})
    ds = hp._maybe_bit_dataset(["x", "y"])
    assert ds.compress is True


def test_highlevel_pipeline_custom_start_id():
    hp = HighLevelPipeline(bit_dataset_params={"start_id": 600})
    ds = hp._maybe_bit_dataset(["a", "b"])
    assert ds.start_id == 600


def test_highlevel_pipeline_duplicate_and_describe():
    hp = HighLevelPipeline()
    hp.add_step(lambda: "a")
    clone = hp.duplicate()
    assert clone.steps == hp.steps
    desc = hp.describe()
    assert "0:" in desc


def test_highlevel_pipeline_run_step_and_execute_until(tmp_path):
    marble_imports.tqdm = std_tqdm
    marble_brain.tqdm = std_tqdm
    marble_main.MetricsVisualizer = MetricsVisualizer

    cfg = _config_path(tmp_path)
    hp = HighLevelPipeline()
    hp.new_marble_system(config_path=str(cfg))
    hp.train_marble_system(train_examples=[(0, 0)], epochs=1)

    marble, first = hp.run_step(0)
    assert isinstance(marble, marble_interface.MARBLE)
    assert first is marble

    marble2, results = hp.execute_until(1)
    assert isinstance(marble2, marble_interface.MARBLE)
    assert len(results) == 2


def test_highlevel_pipeline_insert_and_execute_from(tmp_path):
    marble_imports.tqdm = std_tqdm
    marble_brain.tqdm = std_tqdm
    marble_main.MetricsVisualizer = MetricsVisualizer

    cfg = _config_path(tmp_path)
    hp = HighLevelPipeline()
    hp.new_marble_system(config_path=str(cfg))
    hp.train_marble_system(train_examples=[(0, 0)], epochs=1)

    hp.insert_step(1, lambda marble=None: "x")

    marble, results = hp.execute()
    assert len(results) == 3

    marble2, rest = hp.execute_from(1, marble)
    assert rest[0] == "x"


def test_highlevel_pipeline_replace_and_update():
    hp = HighLevelPipeline()

    def step_a(x=None):
        return x

    hp.add_step(step_a, params={"x": "a"})
    hp.update_step_params(0, x="b")
    assert hp.steps[0]["params"]["x"] == "b"
    hp.replace_step(0, lambda: "c")
    _, result = hp.run_step(0)
    assert result == "c"


def test_highlevel_pipeline_json_methods():
    hp = HighLevelPipeline()
    hp.add_step("new_marble_system", module="marble_interface", params={})
    json_str = hp.to_json()
    clone = HighLevelPipeline.from_json(json_str)
    assert clone.steps == hp.steps


def test_highlevel_pipeline_execute_stream(tmp_path):
    marble_imports.tqdm = std_tqdm
    marble_brain.tqdm = std_tqdm
    marble_main.MetricsVisualizer = MetricsVisualizer

    cfg = _config_path(tmp_path)
    hp = HighLevelPipeline()
    hp.new_marble_system(config_path=str(cfg))
    hp.train_marble_system(train_examples=[(0.0, 0.0)], epochs=1)

    results = list(hp.execute_stream())
    assert isinstance(results[-1][0], marble_interface.MARBLE)
    assert len(results) == 2


def test_highlevel_pipeline_summary_and_clear():
    hp = HighLevelPipeline()
    hp.add_step(lambda: "x")
    info = hp.summary()
    assert info["num_steps"] == 1
    hp.clear_steps()
    assert hp.summary()["num_steps"] == 0


def test_highlevel_pipeline_get_and_list_steps():
    hp = HighLevelPipeline()

    def a():
        return "a"

    hp.add_step(a)
    assert hp.get_step(0)["callable"] == a
    names = hp.list_steps()
    assert names == ["a"]


def test_highlevel_pipeline_execute_range(tmp_path):
    marble_imports.tqdm = std_tqdm
    marble_brain.tqdm = std_tqdm
    marble_main.MetricsVisualizer = MetricsVisualizer

    cfg = _config_path(tmp_path)
    hp = HighLevelPipeline()
    hp.new_marble_system(config_path=str(cfg))
    hp.train_marble_system(train_examples=[(0, 0)], epochs=1)

    marble, results = hp.execute_range(0, 1)
    assert isinstance(marble, marble_interface.MARBLE)
    assert len(results) == 2
