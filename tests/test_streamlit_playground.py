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
import time
from plotly.graph_objs import Figure

from tests.test_core_functions import minimal_params

from streamlit_playground import (
    load_examples,
    initialize_marble,
    list_marble_functions,
    execute_marble_function,
    list_repo_modules,
    list_module_functions,
    list_module_classes,
    create_module_object,
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
    move_pipeline_step,
    remove_pipeline_step,
    pipeline_to_networkx,
    pipeline_figure,
    run_custom_code,
    core_to_networkx,
    core_figure,
    load_yaml_manual,
    set_yaml_value,
    load_hf_model,
    convert_hf_model,
    model_summary,
    search_hf_datasets,
    list_example_projects,
    load_example_code,
    run_example_project,
    start_remote_server,
    create_remote_client,
    create_torrent_system,
    list_learner_modules,
    list_learner_classes,
    create_learner,
    train_learner,
    start_background_training,
    wait_for_training,
    training_in_progress,
    start_auto_firing,
    stop_auto_firing,
    create_gridworld_env,
    run_gridworld_episode,
    metrics_dataframe,
    metrics_figure,
    load_readme,
    load_tutorial,
    search_hf_models,
    core_statistics,
    system_stats,
    get_neuromod_state,
    set_neuromod_state,
    lobe_info,
    add_lobe,
    organize_lobes,
    self_attention_lobes,
    select_high_attention_neurons,
    list_test_files,
    run_tests,
    list_documentation_files,
    load_documentation,
    load_module_source,
    wander_neuronenblitz,
    parallel_wander_neuronenblitz,
    core_weight_matrix,
    core_heatmap_figure,
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


def test_search_hf_datasets():
    dummy = [mock.Mock(id=f"ds{i}") for i in range(3)]
    with mock.patch("huggingface_hub.HfApi") as api:
        inst = api.return_value
        inst.list_datasets.return_value = dummy
        names = search_hf_datasets("cats", limit=3)
    api.assert_called_once()
    inst.list_datasets.assert_called_once_with(search="cats", limit=3)
    assert names == ["ds0", "ds1", "ds2"]


def test_search_hf_models():
    dummy = [mock.Mock(id=f"m{i}") for i in range(2)]
    with mock.patch("huggingface_hub.HfApi") as api:
        inst = api.return_value
        inst.list_models.return_value = dummy
        names = search_hf_models("bert", limit=2)
    api.assert_called_once()
    inst.list_models.assert_called_once_with(search="bert", limit=2)
    assert names == ["m0", "m1"]


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

    dummy_class = types.SimpleNamespace(
        TestClass=type("C", (), {"__init__": lambda self, y=1: setattr(self, "y", y)})
    )
    with mock.patch("importlib.import_module", return_value=dummy_class):
        classes = list_module_classes("dummy")
        assert "TestClass" in classes
        obj = create_module_object("dummy", "TestClass", None, y=2)
        assert obj.y == 2


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


def test_pipeline_move_and_remove():
    pipe = [
        {"func": "a"},
        {"func": "b"},
        {"func": "c"},
    ]
    move_pipeline_step(pipe, 2, 0)
    assert [s["func"] for s in pipe] == ["c", "a", "b"]
    remove_pipeline_step(pipe, 1)
    assert [s["func"] for s in pipe] == ["c", "b"]


def test_pipeline_visualization():
    pipe = [
        {"func": "a"},
        {"func": "b"},
        {"func": "c"},
    ]
    g = pipeline_to_networkx(pipe)
    assert g.number_of_nodes() == 3
    assert g.number_of_edges() == 2
    fig = pipeline_figure(pipe)
    assert hasattr(fig, "to_dict")


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


def test_core_weight_matrix_and_heatmap(tmp_path):
    cfg = {"core": minimal_params(), "brain": {"save_dir": str(tmp_path)}}
    cfg_path = tmp_path / "cfg.yaml"
    with open(cfg_path, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f)
    m = initialize_marble(str(cfg_path))
    mat = core_weight_matrix(m.get_core())
    assert mat.shape[0] == len(m.get_core().neurons)
    assert mat.shape[1] == len(m.get_core().neurons)
    fig = core_heatmap_figure(m.get_core(), limit=2)
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


def test_load_hf_model_and_convert():
    dummy_model = object()
    with mock.patch(
        "streamlit_playground.AutoModel.from_pretrained",
        return_value=dummy_model,
    ) as auto:
        model = load_hf_model("dummy")
    auto.assert_called_once_with("dummy", trust_remote_code=True)
    assert model is dummy_model

    with mock.patch(
        "streamlit_playground.load_hf_model",
        return_value=dummy_model,
    ) as loader, mock.patch(
        "streamlit_playground.marble_interface.convert_pytorch_model",
        return_value="marble",
    ) as conv:
        out = convert_hf_model("dummy")
    loader.assert_called_once_with("dummy")
    conv.assert_called_once()
    assert out == "marble"


def test_model_summary():
    import torch

    model = torch.nn.Linear(2, 1)
    summary = model_summary(model)
    assert "Total parameters" in summary


def test_example_project_helpers():
    projs = list_example_projects()
    assert any(p.startswith("project") for p in projs)

    first = projs[0]
    code = load_example_code(first)
    assert "def" in code

    with mock.patch("streamlit_playground.runpy.run_path") as rp:
        rp.side_effect = lambda path, run_name: print("done")
        out = run_example_project(first)
    rp.assert_called_once()
    assert "done" in out


def test_offloading_helpers(tmp_path):
    server = start_remote_server(port=8010)
    assert server.port == 8010
    client = create_remote_client("http://localhost:8010")
    cfg = {"core": minimal_params(), "brain": {"save_dir": str(tmp_path)}}
    cfg_path = tmp_path / "cfg.yaml"
    with open(cfg_path, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f)
    marble = initialize_marble(str(cfg_path))
    marble.brain.remote_client = client
    marble.brain.offload_enabled = True
    marble.brain.lobe_manager.genesis(range(len(marble.core.neurons)))
    marble.brain.offload_high_attention(threshold=-1.0)
    assert server.core is not None
    server.stop()

    tracker, tclient = create_torrent_system("A", buffer_size=1)
    marble.brain.torrent_client = tclient
    marble.brain.torrent_offload_enabled = True
    marble.brain.offload_high_attention_torrent(threshold=-1.0)
    assert tclient.parts
    tclient.disconnect()


def test_learner_helpers(tmp_path):
    cfg = {"core": minimal_params(), "brain": {"save_dir": str(tmp_path)}}
    cfg_path = tmp_path / "cfg.yaml"
    with open(cfg_path, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f)
    marble = initialize_marble(str(cfg_path))

    modules = list_learner_modules()
    assert "contrastive_learning" in modules
    classes = list_learner_classes("contrastive_learning")
    assert "ContrastiveLearner" in classes

    learner = create_learner("contrastive_learning", "ContrastiveLearner", marble)
    data = [0.0, 1.0]
    train_learner(learner, data, epochs=1)


def test_metrics_and_docs_helpers(tmp_path):
    from marble_base import MetricsVisualizer

    class DummyMarble:
        def __init__(self):
            self.mv = MetricsVisualizer()
            self.mv.metrics["loss"] = [1.0, 0.5]

        def get_metrics_visualizer(self):
            return self.mv

    marble = DummyMarble()
    df = metrics_dataframe(marble)
    fig = metrics_figure(marble)
    assert isinstance(df, pd.DataFrame)
    assert "loss" in df.columns
    assert isinstance(fig, Figure)

    readme = load_readme()
    tutorial = load_tutorial()
    assert "MARBLE" in readme
    assert "Project" in tutorial


def test_system_stats():
    with mock.patch(
        "system_metrics.get_system_memory_usage",
        return_value=123.0,
    ), mock.patch(
        "system_metrics.get_gpu_memory_usage",
        return_value=45.0,
    ):
        stats = system_stats()
    assert stats == {"ram_mb": 123.0, "gpu_mb": 45.0}


def test_core_statistics(tmp_path):
    cfg = {"core": minimal_params(), "brain": {"save_dir": str(tmp_path)}}
    cfg_path = tmp_path / "cfg.yaml"
    with open(cfg_path, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f)
    marble = initialize_marble(str(cfg_path))
    stats = core_statistics(marble)
    assert stats["neurons"] == len(marble.get_core().neurons)
    assert stats["synapses"] == len(marble.get_core().synapses)
    tiers = {n.tier for n in marble.get_core().neurons}
    assert stats["tiers"] == len(tiers)


def test_neuromod_state_helpers(tmp_path):
    cfg = {"core": minimal_params(), "brain": {"save_dir": str(tmp_path)}}
    cfg_path = tmp_path / "cfg.yaml"
    with open(cfg_path, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f)
    marble = initialize_marble(str(cfg_path))

    state = get_neuromod_state(marble)
    assert state["arousal"] == 0.0
    updated = set_neuromod_state(marble, arousal=0.5, emotion="happy")
    assert updated["arousal"] == 0.5
    assert updated["emotion"] == "happy"


def test_async_helpers(tmp_path):
    cfg = {"core": minimal_params(), "brain": {"save_dir": str(tmp_path)}}
    cfg_path = tmp_path / "cfg.yaml"
    with open(cfg_path, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f)
    marble = initialize_marble(str(cfg_path))

    start_background_training(marble, [(0.1, 0.2)], epochs=1)
    timeout = time.time() + 5
    while not training_in_progress(marble) and time.time() < timeout:
        time.sleep(0.01)
    assert training_in_progress(marble)
    wait_for_training(marble)
    assert not training_in_progress(marble)

    start_auto_firing(marble, interval_ms=10)
    time.sleep(0.05)
    assert marble.brain.auto_fire_active
    stop_auto_firing(marble)
    assert not marble.brain.auto_fire_active


def test_list_tests_and_run(tmp_path):
    files = list_test_files()
    assert "test_streamlit_playground.py" in files
    with mock.patch("streamlit_playground.pytest.main", return_value=0) as pm:
        out = run_tests("dummy")
    pm.assert_called_once_with(["-k", "dummy"])
    assert "Exit code" in out


def test_documentation_helpers():
    docs = list_documentation_files()
    assert "README.md" in docs
    text = load_documentation("README.md")
    assert "MARBLE" in text


def test_source_browser():
    code = load_module_source("reinforcement_learning")
    assert "def" in code


def test_gridworld_helpers(tmp_path):
    cfg = {"core": minimal_params(), "brain": {"save_dir": str(tmp_path)}}
    cfg_path = tmp_path / "cfg.yaml"
    with open(cfg_path, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f)
    marble = initialize_marble(str(cfg_path))
    env = create_gridworld_env(size=3)
    assert env.size == 3
    rewards = run_gridworld_episode(marble, episodes=1, max_steps=2, size=3)
    assert isinstance(rewards, list) and len(rewards) == 1


def test_lobe_manager_helpers(tmp_path):
    cfg = {"core": minimal_params(), "brain": {"save_dir": str(tmp_path)}}
    cfg_path = tmp_path / "cfg.yaml"
    with open(cfg_path, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f)
    marble = initialize_marble(str(cfg_path))

    assert lobe_info(marble) == []
    lid = add_lobe(marble, [0, 1])
    assert isinstance(lid, int)
    info = lobe_info(marble)
    assert len(info) == 1 and info[0]["neurons"] == 2
    count = organize_lobes(marble)
    assert isinstance(count, int)
    self_attention_lobes(marble, loss=1.0)
    ids = select_high_attention_neurons(marble, threshold=-1.0)
    assert isinstance(ids, list)


def test_neuronenblitz_wander_helpers(tmp_path):
    cfg = {"core": minimal_params(), "brain": {"save_dir": str(tmp_path)}}
    cfg_path = tmp_path / "cfg.yaml"
    with open(cfg_path, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f)
    marble = initialize_marble(str(cfg_path))

    out, path = wander_neuronenblitz(marble, 0.1)
    assert isinstance(out, float)
    assert isinstance(path, list)

    res = parallel_wander_neuronenblitz(marble, 0.2, processes=1)
    assert isinstance(res, list) and res
    assert isinstance(res[0][0], float)
    assert isinstance(res[0][1], int)
