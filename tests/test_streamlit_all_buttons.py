import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import types
import streamlit_playground as sp
from streamlit.testing.v1 import AppTest


def _setup(monkeypatch):
    monkeypatch.setattr(sp, "train_marble_system", lambda *a, **k: None)
    monkeypatch.setattr(sp, "load_marble_system", lambda *a, **k: sp.new_marble_system("config.yaml"))
    monkeypatch.setattr(sp, "save_marble_system", lambda *a, **k: None)
    monkeypatch.setattr(sp, "export_core_to_json", lambda m: "{}")
    monkeypatch.setattr(sp, "import_core_from_json", lambda js: sp.new_marble_system("config.yaml"))
    monkeypatch.setattr(sp, "load_hf_examples", lambda *a, **k: [(0,0)])
    monkeypatch.setattr(sp, "search_hf_datasets", lambda *a, **k: [])
    monkeypatch.setattr(sp, "search_hf_models", lambda *a, **k: [])
    monkeypatch.setattr(sp, "load_hf_model", lambda *a, **k: types.SimpleNamespace(named_parameters=lambda: []))
    monkeypatch.setattr(sp, "convert_hf_model", lambda *a, **k: sp.new_marble_system("config.yaml"))
    monkeypatch.setattr(sp, "start_metrics_dashboard", lambda *a, **k: types.SimpleNamespace(stop=lambda: None))
    monkeypatch.setattr(sp, "start_background_training", lambda *a, **k: None)
    monkeypatch.setattr(sp, "wait_for_training", lambda *a, **k: None)
    monkeypatch.setattr(sp, "start_auto_firing", lambda *a, **k: None)
    monkeypatch.setattr(sp, "stop_auto_firing", lambda *a, **k: None)
    monkeypatch.setattr(sp, "run_gridworld_episode", lambda *a, **k: [])
    monkeypatch.setattr(sp, "update_meta_controller", lambda *a, **k: {})
    monkeypatch.setattr(sp, "adjust_meta_controller", lambda *a, **k: 0)
    monkeypatch.setattr(sp, "reset_meta_loss_history", lambda *a, **k: None)
    monkeypatch.setattr(sp, "super_evo_history", lambda *a, **k: [])
    monkeypatch.setattr(sp, "super_evo_changes", lambda *a, **k: [])
    monkeypatch.setattr(sp, "clear_super_evo_changes", lambda *a, **k: None)
    monkeypatch.setattr(sp, "run_dimensional_search", lambda *a, **k: 0)
    monkeypatch.setattr(sp, "run_nd_topology", lambda *a, **k: 0)
    monkeypatch.setattr(sp, "create_hybrid_memory", lambda *a, **k: None)
    monkeypatch.setattr(sp, "hybrid_memory_store", lambda *a, **k: None)
    monkeypatch.setattr(sp, "hybrid_memory_retrieve", lambda *a, **k: [])
    monkeypatch.setattr(sp, "hybrid_memory_forget", lambda *a, **k: None)
    monkeypatch.setattr(sp, "wander_neuronenblitz", lambda *a, **k: (0, []))
    monkeypatch.setattr(sp, "parallel_wander_neuronenblitz", lambda *a, **k: [])
    monkeypatch.setattr(sp, "run_example_project", lambda *a, **k: "")
    monkeypatch.setattr(sp, "run_tests", lambda *a, **k: "")
    monkeypatch.setattr(sp, "load_module_source", lambda *a, **k: "")

    at = AppTest.from_file("streamlit_playground.py").run(timeout=10)
    at = at.sidebar.button[0].click().run(timeout=10)
    at = at.sidebar.radio[0].set_value("Advanced").run(timeout=10)
    return at


def test_click_all_buttons(monkeypatch):
    at = _setup(monkeypatch)
    buttons = list(at.sidebar.button)
    for tab in at.tabs:
        buttons.extend(tab.button)
    assert buttons, "No buttons found"
    for b in buttons:
        at = b.click().run(timeout=1)
