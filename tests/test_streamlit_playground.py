import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import yaml
from unittest import mock

import pandas as pd
from PIL import Image
from zipfile import ZipFile
from io import BytesIO
import numpy as np

from tests.test_core_functions import minimal_params

from streamlit_playground import (
    load_examples,
    initialize_marble,
    list_marble_functions,
    execute_marble_function,
    list_repo_modules,
    list_module_functions,
    execute_module_function,
    execute_function_sequence,
    save_marble_system,
    load_marble_system,
    export_core_to_json,
    import_core_from_json,
    load_hf_examples,
    start_metrics_dashboard,
    preview_file_dataset,
    preview_hf_dataset,
    save_pipeline_to_json,
    load_pipeline_from_json,
    run_custom_code,
    core_to_networkx,
    core_figure,
    load_yaml_manual,
    set_yaml_value,
)


def test_load_examples_csv_and_json(tmp_path):
    csv_path = tmp_path / "data.csv"
    csv_path.write_text("input,target\n0.1,0.2\n0.2,0.4\n")
    with open(csv_path, "r", encoding="utf-8") as f:
        ex = load_examples(f)
    assert ex == [(0.1, 0.2), (0.2, 0.4)]

    json_path = tmp_path / "data.json"
    pd.DataFrame({"input": [1, 2], "target": [2, 4]}).to_json(json_path)
    with open(json_path, "r", encoding="utf-8") as f:
        ex = load_examples(f)
    assert ex == [(1.0, 2.0), (2.0, 4.0)]


def test_load_examples_zip(tmp_path):
    img = Image.new("RGB", (2, 2), color=(255, 0, 0))
    buf = BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)

    zip_path = tmp_path / "data.zip"
    with ZipFile(zip_path, "w") as zf:
        zf.writestr("inputs/img1.png", buf.getvalue())
        zf.writestr("targets/img1.png", buf.getvalue())

    with open(zip_path, "rb") as f:
        ex = load_examples(f)
    assert len(ex) == 1
    assert isinstance(ex[0][0], np.ndarray)


def test_initialize_marble(tmp_path):
    cfg = {"core": minimal_params(), "brain": {"save_dir": str(tmp_path)}}
    cfg_path = tmp_path / "cfg.yaml"
    with open(cfg_path, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f)
    m = initialize_marble(str(cfg_path))
    assert len(m.get_core().neurons) > 0


def test_initialize_marble_with_yaml_text(tmp_path):
    cfg = {"core": minimal_params(), "brain": {"save_dir": str(tmp_path)}}
    yaml_str = yaml.dump(cfg)
    m = initialize_marble(None, yaml_text=yaml_str)
    assert len(m.get_core().neurons) > 0


def test_execute_marble_function(tmp_path):
    cfg = {"core": minimal_params(), "brain": {"save_dir": str(tmp_path)}}
    cfg_path = tmp_path / "cfg.yaml"
    with open(cfg_path, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f)
    m = initialize_marble(str(cfg_path))
    funcs = list_marble_functions()
    assert "count_marble_synapses" in funcs
    out = execute_marble_function("count_marble_synapses", m)
    assert isinstance(out, int)


def test_save_load_and_core_roundtrip(tmp_path):
    cfg = {"core": minimal_params(), "brain": {"save_dir": str(tmp_path)}}
    cfg_path = tmp_path / "cfg.yaml"
    with open(cfg_path, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f)
    m = initialize_marble(str(cfg_path))

    save_path = tmp_path / "model.pkl"
    save_marble_system(m, str(save_path))
    assert save_path.exists()

    m2 = load_marble_system(str(save_path))
    assert len(m2.get_core().neurons) == len(m.get_core().neurons)

    js = export_core_to_json(m)
    m3 = import_core_from_json(js)
    assert len(m3.get_core().neurons) == len(m.get_core().neurons)


def test_load_hf_examples_and_dashboard(tmp_path):
    with mock.patch(
        "streamlit_playground.load_hf_dataset",
        return_value=[(0.1, 0.2), (0.2, 0.4)],
    ):
        ex = load_hf_examples("dummy", "train")
    assert ex == [(0.1, 0.2), (0.2, 0.4)]

    class DummyMarble:
        def __init__(self):
            from marble_base import MetricsVisualizer

            self.mv = MetricsVisualizer()

        def get_metrics_visualizer(self):
            return self.mv

    with mock.patch("streamlit_playground.MetricsDashboard.start") as start:
        dash = start_metrics_dashboard(DummyMarble(), port=8062)
        start.assert_called_once()
        assert dash.port == 8062


def test_dataset_previews(tmp_path):
    csv_path = tmp_path / "data.csv"
    csv_path.write_text("input,target\n1,2\n3,4\n")
    with open(csv_path, "rb") as f:
        df = preview_file_dataset(f)
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["input", "target"]
    assert df.shape[0] == 2

    with mock.patch(
        "streamlit_playground.load_hf_examples",
        return_value=[(0.1, 0.2), (0.2, 0.4)],
    ):
        df2 = preview_hf_dataset("dummy", "train")
    assert isinstance(df2, pd.DataFrame)
    assert df2.shape == (2, 2)


def test_module_listing_and_execution():
    mods = list_repo_modules()
    assert "reinforcement_learning" in mods
    funcs = list_module_functions("reinforcement_learning")
    assert "train_gridworld" in funcs

    import types

    dummy = types.SimpleNamespace(test_func=lambda x: x + 1)
    with mock.patch("importlib.import_module", return_value=dummy):
        out = execute_module_function("dummy", "test_func", x=1)
    assert out == 2


def test_execute_function_sequence(tmp_path):
    cfg = {"core": minimal_params(), "brain": {"save_dir": str(tmp_path)}}
    cfg_path = tmp_path / "cfg.yaml"
    with open(cfg_path, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f)
    m = initialize_marble(str(cfg_path))

    steps = [
        {"func": "count_marble_synapses"},
        {
            "module": "reinforcement_learning",
            "func": "train_gridworld",
            "params": {"episodes": 1},
        },
    ]

    with mock.patch(
        "streamlit_playground.execute_module_function",
        return_value="ok",
    ) as mod_exec:
        results = execute_function_sequence(steps, m)
    assert isinstance(results, list)
    assert len(results) == 2
    mod_exec.assert_called_once()


def test_pipeline_save_load(tmp_path):
    pipeline = [{"func": "count_marble_synapses"}]
    path = tmp_path / "pipe.json"
    save_pipeline_to_json(pipeline, path)
    assert path.exists()
    with open(path, "r", encoding="utf-8") as f:
        loaded = load_pipeline_from_json(f)
    assert loaded == pipeline


def test_run_custom_code(tmp_path):
    cfg = {"core": minimal_params(), "brain": {"save_dir": str(tmp_path)}}
    cfg_path = tmp_path / "cfg.yaml"
    with open(cfg_path, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f)
    m = initialize_marble(str(cfg_path))
    code = "result = len(marble.get_core().neurons)"
    out = run_custom_code(code, m)
    assert out == len(m.get_core().neurons)


def test_core_network_visualization(tmp_path):
    cfg = {"core": minimal_params(), "brain": {"save_dir": str(tmp_path)}}
    cfg_path = tmp_path / "cfg.yaml"
    with open(cfg_path, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f)
    m = initialize_marble(str(cfg_path))
    g = core_to_networkx(m.get_core())
    assert g.number_of_nodes() == len(m.get_core().neurons)
    fig = core_figure(m.get_core())
    assert hasattr(fig, "to_dict")


def test_load_yaml_manual_text():
    text = load_yaml_manual()
    assert "core:" in text


def test_set_yaml_value_simple():
    yaml_text = "a:\n  b: 1\n"
    new = set_yaml_value(yaml_text, "a.b", 2)
    cfg = yaml.safe_load(new)
    assert cfg["a"]["b"] == 2


def test_set_yaml_value_nested_creation():
    yaml_text = "a: {}\n"
    new = set_yaml_value(yaml_text, "a.c.d", 5)
    cfg = yaml.safe_load(new)
    assert cfg["a"]["c"]["d"] == 5
