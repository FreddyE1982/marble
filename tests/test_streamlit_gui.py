import os
import sys
import warnings

from _pytest.warning_types import PytestDeprecationWarning
from streamlit.testing.v1 import AppTest

# Suppress protobuf deprecation warnings from dependencies before importing
warnings.filterwarnings(
    "ignore",
    message=".*PyType_Spec.*",
    category=DeprecationWarning,
)

warnings.filterwarnings(
    "ignore",
    category=PytestDeprecationWarning,
    module="dash.testing.plugin",
)

# Ensure repo root is on path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def test_playground_initial_load():
    at = AppTest.from_file("streamlit_playground.py").run(timeout=15)
    assert at.title[0].value == "MARBLE Playground"
    assert any("Initialize MARBLE" in info.value for info in at.info)


def test_playground_mode_switching():
    at = AppTest.from_file("streamlit_playground.py").run(timeout=15)

    # create a MARBLE instance using the sidebar button
    at = at.sidebar.button[0].click().run(timeout=30)
    assert any("created" in s.value for s in at.sidebar.success)

    # a radio selector should appear for choosing the interface mode
    assert len(at.sidebar.radio) == 1
    assert at.sidebar.radio[0].value == "Basic"

    # basic mode does not display the advanced tabs
    assert len(at.tabs) == 0

    # switch to advanced mode and verify tabs exist
    at = at.sidebar.radio[0].set_value("Advanced").run(timeout=20)
    labels = [t.label for t in at.tabs]
    assert "Function Search" in labels
    assert "marble_interface" in labels
    assert len(labels) >= 20


def _setup_advanced_playground(timeout: float = 20) -> AppTest:
    """Return an ``AppTest`` instance with MARBLE initialized in advanced mode."""
    at = AppTest.from_file("streamlit_playground.py").run(timeout=15)
    at = at.sidebar.button[0].click().run(timeout=30)
    return at.sidebar.radio[0].set_value("Advanced").run(timeout=timeout)


def _setup_basic_playground(timeout: float = 20) -> AppTest:
    """Return an ``AppTest`` instance with MARBLE initialized in basic mode."""
    at = AppTest.from_file("streamlit_playground.py").run(timeout=15)
    return at.sidebar.button[0].click().run(timeout=timeout)


def test_stats_tab_metrics():
    at = _setup_advanced_playground()
    stats_tab = next(t for t in at.tabs if t.label == "System Stats")

    labels = [m.label for m in stats_tab.metric]
    assert "RAM" in labels and "GPU" in labels

    # refresh should rerun without removing metrics
    at = stats_tab.button[0].click().run(timeout=20)
    stats_tab = next(t for t in at.tabs if t.label == "System Stats")
    assert len(stats_tab.metric) == 2


def test_metrics_tab_plot():
    at = _setup_advanced_playground()
    metrics_tab = next(t for t in at.tabs if t.label == "Metrics")

    assert metrics_tab.get("plotly_chart"), "metrics plot not found"
    assert metrics_tab.button and metrics_tab.button[0].label == "Refresh"

    at = metrics_tab.button[0].click().run(timeout=20)
    metrics_tab = next(t for t in at.tabs if t.label == "Metrics")
    assert metrics_tab.get("plotly_chart")


def test_neuromod_tab_update():
    at = _setup_advanced_playground()
    neuro_tab = next(t for t in at.tabs if t.label == "Neuromodulation")

    neuro_tab.slider[0].set_value(0.5)
    neuro_tab.slider[1].set_value(0.4)
    neuro_tab.slider[2].set_value(0.3)
    neuro_tab.text_input[0].input("excited")
    at = neuro_tab.button[0].click().run(timeout=20)
    neuro_tab = next(t for t in at.tabs if t.label == "Neuromodulation")
    assert any("Signals updated" in s.value for s in neuro_tab.success)


def test_config_editor_update(tmp_path):
    at = _setup_advanced_playground()
    cfg_tab = next(t for t in at.tabs if t.label == "Config Editor")

    cfg_tab.text_input[0].input("core.width")
    cfg_tab.text_input[1].input("5")
    at = cfg_tab.button[0].click().run(timeout=20)
    cfg_tab = next(t for t in at.tabs if t.label == "Config Editor")
    assert cfg_tab.code and "core:" in cfg_tab.code[0].value


def test_hybrid_memory_store_and_retrieve():
    at = _setup_advanced_playground()
    hm_tab = next(t for t in at.tabs if t.label == "Hybrid Memory")

    hm_tab.text_input[0].input("k")
    hm_tab.text_input[1].input("1.0")
    at = hm_tab.button[0].click().run(timeout=20)
    hm_tab = next(t for t in at.tabs if t.label == "Hybrid Memory")
    assert any("Stored" in s.value for s in hm_tab.success)

    hm_tab.text_input[2].input("1.0")
    hm_tab.number_input[0].set_value(1)
    at = hm_tab.button[1].click().run(timeout=20)
    hm_tab = next(t for t in at.tabs if t.label == "Hybrid Memory")
    assert hm_tab.markdown


def test_visualization_and_heatmap_tabs():
    at = _setup_advanced_playground()
    vis_tab = next(t for t in at.tabs if t.label == "Visualization")
    vis_tab.selectbox[0].set_value("circular")
    vis_tab.button[0].click()
    at = vis_tab.run(timeout=20)
    vis_tab = next(t for t in at.tabs if t.label == "Visualization")
    assert vis_tab.get("plotly_chart")

    heat_tab = next(t for t in at.tabs if t.label == "Weight Heatmap")
    heat_tab.number_input[0].set_value(10)
    heat_tab.selectbox[0].set_value("Plasma")
    heat_tab.button[0].click()
    at = heat_tab.run(timeout=20)
    heat_tab = next(t for t in at.tabs if t.label == "Weight Heatmap")
    assert heat_tab.get("plotly_chart")


def test_documentation_tab_view():
    at = _setup_advanced_playground()
    docs_tab = next(t for t in at.tabs if t.label == "Documentation")
    docs_tab.selectbox[0].set_value("README.md")
    at = docs_tab.run(timeout=20)
    docs_tab = next(t for t in at.tabs if t.label == "Documentation")
    assert docs_tab.code and "MARBLE" in docs_tab.code[0].value


def test_function_search_count_synapses():
    at = _setup_advanced_playground()
    search_tab = next(t for t in at.tabs if t.label == "Function Search")
    search_tab.text_input[0].input("count_marble_synapses")
    at = search_tab.run(timeout=20)
    search_tab = next(t for t in at.tabs if t.label == "Function Search")
    assert search_tab.selectbox
    search_tab.selectbox[0].set_value("marble_interface.count_marble_synapses")
    at = search_tab.button[0].click().run(timeout=20)
    search_tab = next(t for t in at.tabs if t.label == "Function Search")
    assert any(md.value.strip("`").isdigit() for md in search_tab.markdown)


def test_interface_tab_exec_function():
    at = _setup_advanced_playground()
    iface_tab = next(t for t in at.tabs if t.label == "marble_interface")
    iface_tab.text_input[0].input("count_marble_synapses")
    at = iface_tab.run(timeout=20)
    iface_tab = next(t for t in at.tabs if t.label == "marble_interface")
    iface_tab.selectbox[0].set_value("count_marble_synapses")
    at = iface_tab.button[0].click().run(timeout=20)
    iface_tab = next(t for t in at.tabs if t.label == "marble_interface")
    assert any(md.value.strip("`").isdigit() for md in iface_tab.markdown)


def test_modules_tab_exec_function():
    at = _setup_advanced_playground()
    mod_tab = next(t for t in at.tabs if t.label == "Modules")
    mod_tab.selectbox[0].set_value("marble_interface")
    mod_tab.text_input[0].input("count_marble_synapses")
    at = mod_tab.run(timeout=20)
    mod_tab = next(t for t in at.tabs if t.label == "Modules")
    mod_tab.selectbox[1].set_value("count_marble_synapses")
    at = mod_tab.button[0].click().run(timeout=20)
    mod_tab = next(t for t in at.tabs if t.label == "Modules")
    assert any(md.value.strip("`").isdigit() for md in mod_tab.markdown)


def test_classes_tab_instantiation():
    at = _setup_advanced_playground()
    cls_tab = next(t for t in at.tabs if t.label == "Classes")
    cls_tab.selectbox[0].set_value("marble_registry")
    at = cls_tab.run(timeout=20)
    cls_tab = next(t for t in at.tabs if t.label == "Classes")
    cls_tab.selectbox[1].set_value("MarbleRegistry")
    at = cls_tab.button[0].click().run(timeout=20)
    cls_tab = next(t for t in at.tabs if t.label == "Classes")
    assert any("Created MarbleRegistry" in s.value for s in cls_tab.success)


def test_pipeline_tab_add_and_run():
    at = _setup_advanced_playground()
    pipe_tab = next(t for t in at.tabs if t.label == "Pipeline")

    add_expander = pipe_tab.expander[1]
    add_expander.selectbox[0].set_value("count_marble_synapses")
    at = add_expander.button[0].click().run(timeout=20)
    pipe_tab = next(t for t in at.tabs if t.label == "Pipeline")
    assert pipe_tab.button[3].label == "Run Pipeline"
    at = pipe_tab.button[3].click().run(timeout=20)
    pipe_tab = next(t for t in at.tabs if t.label == "Pipeline")
    assert any(md.value.strip("`").isdigit() for md in pipe_tab.markdown)


def test_custom_code_execution():
    at = _setup_advanced_playground()
    code_tab = next(t for t in at.tabs if t.label == "Custom Code")
    code_tab.text_area[0].input("result = 123")
    at = code_tab.button[0].click().run(timeout=20)
    code_tab = next(t for t in at.tabs if t.label == "Custom Code")
    assert any(md.value.strip("`").isdigit() for md in code_tab.markdown)


def test_rl_sandbox_run():
    at = _setup_advanced_playground()
    rl_tab = next(t for t in at.tabs if t.label == "RL Sandbox")
    rl_tab.number_input[0].set_value(3)
    rl_tab.number_input[1].set_value(1)
    rl_tab.number_input[2].set_value(2)
    at = rl_tab.button[0].click().run(timeout=20)
    rl_tab = next(t for t in at.tabs if t.label == "RL Sandbox")
    assert rl_tab.get("plotly_chart")


def test_async_autofire_start_stop():
    at = _setup_advanced_playground()
    async_tab = next(t for t in at.tabs if t.label == "Async Training")
    at = async_tab.button[2].click().run(timeout=20)
    async_tab = next(t for t in at.tabs if t.label == "Async Training")
    assert any("Auto-firing started" in s.value for s in async_tab.success)
    at = async_tab.button[3].click().run(timeout=20)
    async_tab = next(t for t in at.tabs if t.label == "Async Training")
    assert any("Auto-firing stopped" in s.value for s in async_tab.success)


def test_async_training_start_and_wait(monkeypatch):
    monkeypatch.setattr(
        "streamlit_playground.start_background_training", lambda *a, **k: None
    )
    monkeypatch.setattr("streamlit_playground.wait_for_training", lambda *a, **k: None)
    at = _setup_advanced_playground()
    at.session_state["hf_examples"] = [(1, 2)]
    async_tab = next(t for t in at.tabs if t.label == "Async Training")

    at = async_tab.button[0].click().run(timeout=20)
    async_tab = next(t for t in at.tabs if t.label == "Async Training")
    assert any("Training started" in s.value for s in async_tab.success)

    at = async_tab.button[1].click().run(timeout=20)
    async_tab = next(t for t in at.tabs if t.label == "Async Training")
    assert any("Training complete" in s.value for s in async_tab.success)


def test_lobe_manager_create_and_select():
    at = _setup_advanced_playground()
    lobe_tab = next(t for t in at.tabs if t.label == "Lobe Manager")

    exp = lobe_tab.expander[0]
    exp.text_input[0].input("0,1")
    at = exp.button[0].click().run(timeout=20)
    lobe_tab = next(t for t in at.tabs if t.label == "Lobe Manager")
    assert any("created" in s.value.lower() for s in lobe_tab.success)


def test_core_tools_expand():
    at = _setup_advanced_playground()
    core_tab = next(t for t in at.tabs if t.label == "Core Tools")

    core_tab.number_input[0].set_value(1)
    core_tab.number_input[1].set_value(1)
    at = core_tab.button[0].click().run(timeout=20)
    core_tab = next(t for t in at.tabs if t.label == "Core Tools")
    assert any("core expanded" in s.value.lower() for s in core_tab.success)


def test_learner_creation():
    at = _setup_advanced_playground()
    learn_tab = next(t for t in at.tabs if t.label == "Learners")

    learn_tab.selectbox[0].set_value("hebbian_learning")
    at = learn_tab.run(timeout=20)
    learn_tab = next(t for t in at.tabs if t.label == "Learners")
    learn_tab.selectbox[1].set_value("HebbianLearner")
    at = learn_tab.button[0].click().run(timeout=20)
    learn_tab = next(t for t in at.tabs if t.label == "Learners")
    assert any("learner created" in s.value.lower() for s in learn_tab.success)


def test_projects_tab_show_code():
    at = _setup_advanced_playground()
    proj_tab = next(t for t in at.tabs if t.label == "Projects")

    proj_tab.selectbox[0].set_value("project05_gpt_training.py")
    at = proj_tab.run(timeout=20)
    proj_tab = next(t for t in at.tabs if t.label == "Projects")
    assert proj_tab.expander[0].code and "import" in proj_tab.expander[0].code[0].value


def test_model_conversion_preview(monkeypatch):
    import torch

    monkeypatch.setattr(
        "transformers.AutoModel.from_pretrained",
        lambda *a, **k: torch.nn.Linear(1, 1),
    )
    at = _setup_advanced_playground()
    model_tab = next(t for t in at.tabs if t.label == "Model Conversion")

    model_tab.text_input[1].input("dummy")
    at = model_tab.button[1].click().run(timeout=20)
    model_tab = next(t for t in at.tabs if t.label == "Model Conversion")
    assert any("Total parameters" in t.value for t in model_tab.text)


def test_offloading_remote_server(monkeypatch):
    dummy = type("Srv", (), {"stop": lambda self: None})()
    monkeypatch.setattr("streamlit_playground.start_remote_server", lambda **kw: dummy)
    at = _setup_advanced_playground()
    off_tab = next(t for t in at.tabs if t.label == "Offloading")

    exp = off_tab.expander[0]
    at = exp.button[0].click().run(timeout=20)
    off_tab = next(t for t in at.tabs if t.label == "Offloading")
    assert any("Server started" in s.value for s in off_tab.success)

    exp = off_tab.expander[0]
    at = exp.button[1].click().run(timeout=20)
    off_tab = next(t for t in at.tabs if t.label == "Offloading")
    assert any("Server stopped" in s.value for s in off_tab.success)


def test_offloading_client_and_torrent(monkeypatch):
    client = object()
    monkeypatch.setattr("streamlit_playground.create_remote_client", lambda url: client)
    tclient = type("TC", (), {"disconnect": lambda self: None})()
    monkeypatch.setattr(
        "streamlit_playground.create_torrent_system",
        lambda *a, **k: (object(), tclient),
    )
    at = _setup_advanced_playground()
    off_tab = next(t for t in at.tabs if t.label == "Offloading")

    client_exp = off_tab.expander[1]
    at = client_exp.button[0].click().run(timeout=20)
    off_tab = next(t for t in at.tabs if t.label == "Offloading")
    assert any("Client created" in s.value for s in off_tab.success)

    client_exp = off_tab.expander[1]
    at = client_exp.button[1].click().run(timeout=20)
    off_tab = next(t for t in at.tabs if t.label == "Offloading")
    assert any("Client attached" in s.value for s in off_tab.success)

    tor_exp = off_tab.expander[2]
    at = tor_exp.button[0].click().run(timeout=20)
    off_tab = next(t for t in at.tabs if t.label == "Offloading")
    assert any("Torrent client started" in s.value for s in off_tab.success)

    tor_exp = off_tab.expander[2]
    at = tor_exp.button[1].click().run(timeout=20)
    off_tab = next(t for t in at.tabs if t.label == "Offloading")
    assert any("Torrent client stopped" in s.value for s in off_tab.success)


def test_adaptive_control_update(monkeypatch):
    monkeypatch.setattr(
        "streamlit_playground.update_meta_controller",
        lambda *a, **k: {"history_length": 1},
    )
    at = _setup_advanced_playground()
    adapt_tab = next(t for t in at.tabs if t.label == "Adaptive Control")
    exp = adapt_tab.expander[0]
    exp.number_input[0].set_value(2)
    at = exp.button[0].click().run(timeout=20)
    adapt_tab = next(t for t in at.tabs if t.label == "Adaptive Control")
    assert adapt_tab.json


def test_nb_explorer_wander(monkeypatch):
    monkeypatch.setattr(
        "streamlit_playground.wander_neuronenblitz",
        lambda *a, **k: (0.1, [1, 2]),
    )
    at = _setup_advanced_playground()
    nb_tab = next(t for t in at.tabs if t.label == "NB Explorer")
    nb_tab.text_input[0].input("0.5")
    at = nb_tab.button[0].click().run(timeout=20)
    nb_tab = next(t for t in at.tabs if t.label == "NB Explorer")
    assert any("Output" in md.value for md in nb_tab.markdown)


def test_nb_explorer_parallel_and_history(monkeypatch):
    monkeypatch.setattr(
        "streamlit_playground.parallel_wander_neuronenblitz",
        lambda *a, **k: [(0.2, 1)],
    )

    at = _setup_advanced_playground()
    nb = at.session_state["marble"].get_neuronenblitz()
    monkeypatch.setattr(nb, "get_training_history", lambda: [{"step": 1, "loss": 0.5}])

    nb_tab = next(t for t in at.tabs if t.label == "NB Explorer")
    nb_tab.text_input[0].input("0.5")
    at = nb_tab.button[1].click().run(timeout=20)
    nb_tab = next(t for t in at.tabs if t.label == "NB Explorer")
    at = nb_tab.button[2].click().run(timeout=20)
    nb_tab = next(t for t in at.tabs if t.label == "NB Explorer")
    assert nb_tab.button[1].label == "Parallel Wander"
    assert nb_tab.button[2].label == "Show Training History"


def test_tests_tab_run(monkeypatch):
    monkeypatch.setattr(
        "streamlit_playground.run_tests", lambda pattern=None: "Exit code: 0"
    )
    at = _setup_advanced_playground()
    tests_tab = next(t for t in at.tabs if t.label == "Tests")
    tests_tab.multiselect[0].set_value(["test_streamlit_gui.py"])
    at = tests_tab.button[0].click().run(timeout=20)
    tests_tab = next(t for t in at.tabs if t.label == "Tests")
    assert any("Exit code" in t.value for t in tests_tab.text)


def test_source_browser_show(monkeypatch):
    monkeypatch.setattr(
        "streamlit_playground.load_module_source", lambda mod: "def x(): pass"
    )
    at = _setup_advanced_playground()
    src_tab = next(t for t in at.tabs if t.label == "Source Browser")
    src_tab.selectbox[0].set_value("marble_interface")
    at = src_tab.button[0].click().run(timeout=20)
    src_tab = next(t for t in at.tabs if t.label == "Source Browser")
    assert any("def x" in code.value for code in src_tab.code)


def test_pipeline_tab_graph_and_clear():
    at = _setup_advanced_playground()
    pipe_tab = next(t for t in at.tabs if t.label == "Pipeline")

    add_expander = pipe_tab.expander[1]
    add_expander.selectbox[0].set_value("count_marble_synapses")
    at = add_expander.button[0].click().run(timeout=20)
    pipe_tab = next(t for t in at.tabs if t.label == "Pipeline")

    graph_btn = next(b for b in pipe_tab.button if b.label == "Show Pipeline Graph")
    at = graph_btn.click().run(timeout=20)
    pipe_tab = next(t for t in at.tabs if t.label == "Pipeline")
    assert pipe_tab.get("plotly_chart")

    run_btn = next(b for b in pipe_tab.button if b.label == "Run Pipeline")
    at = run_btn.click().run(timeout=20)
    pipe_tab = next(t for t in at.tabs if t.label == "Pipeline")
    clear_btn = next(b for b in pipe_tab.button if b.label == "Clear Pipeline")
    at = clear_btn.click().run(timeout=20)
    assert at.session_state["pipeline"] == []


def test_pipeline_step_visualisation():
    at = _setup_advanced_playground()
    pipe_tab = next(t for t in at.tabs if t.label == "Pipeline")

    add_expander = pipe_tab.expander[1]
    add_expander.selectbox[0].set_value("increase_marble_representation")
    add_expander.number_input[0].set_value(2)
    at = add_expander.button[0].click().run(timeout=20)
    pipe_tab = next(t for t in at.tabs if t.label == "Pipeline")

    vis_exp = next(e for e in pipe_tab.expander if e.label == "Step Visualisation")
    assert any("delta" in j.value for j in vis_exp.json)


def _setup_pipeline_for_export(device: str) -> tuple[AppTest, any]:
    at = AppTest.from_file("streamlit_playground.py")
    at.query_params["device"] = device
    at = at.run(timeout=15)
    at = at.sidebar.button[0].click().run(timeout=30)
    at = at.sidebar.radio[0].set_value("Advanced").run(timeout=20)
    pipe_tab = next(t for t in at.tabs if t.label == "Pipeline")
    add_exp = pipe_tab.expander[1]
    add_exp.selectbox[0].set_value("increase_marble_representation")
    add_exp.number_input[0].set_value(2)
    at = add_exp.button[0].click().run(timeout=20)
    pipe_tab = next(t for t in at.tabs if t.label == "Pipeline")
    return at, pipe_tab


def test_step_export_and_metrics_desktop():
    at, pipe_tab = _setup_pipeline_for_export("desktop")
    vis_exp = next(e for e in pipe_tab.expander if e.label == "Step Visualisation")
    labels = [b.label for b in vis_exp.download_button]
    assert "Download JSON" in labels and "Download CSV" in labels
    run_btn = next(b for b in pipe_tab.button if b.label == "Run Pipeline")
    at = run_btn.click().run(timeout=20)
    pipe_tab = next(t for t in at.tabs if t.label == "Pipeline")
    vis_exp = next(e for e in pipe_tab.expander if e.label == "Step Visualisation")
    assert vis_exp.metric, "metric panels missing"


def test_step_export_and_metrics_mobile():
    at, pipe_tab = _setup_pipeline_for_export("mobile")
    vis_exp = next(e for e in pipe_tab.expander if e.label == "Step Visualisation")
    labels = [b.label for b in vis_exp.download_button]
    assert "Download JSON" in labels and "Download CSV" in labels
    run_btn = next(b for b in pipe_tab.button if b.label == "Run Pipeline")
    at = run_btn.click().run(timeout=20)
    pipe_tab = next(t for t in at.tabs if t.label == "Pipeline")
    vis_exp = next(e for e in pipe_tab.expander if e.label == "Step Visualisation")
    assert vis_exp.metric, "metric panels missing"


def test_lobe_manager_actions():
    at = _setup_advanced_playground()
    lobe_tab = next(t for t in at.tabs if t.label == "Lobe Manager")

    org_btn = next(b for b in lobe_tab.button if b.label == "Organize Lobes")
    at = org_btn.click().run(timeout=20)
    lobe_tab = next(t for t in at.tabs if t.label == "Lobe Manager")
    assert any("organized" in s.value.lower() for s in lobe_tab.success)

    lobe_tab.number_input[0].set_value(0.1)
    sa_btn = next(b for b in lobe_tab.button if b.label == "Apply Self-Attention")
    at = sa_btn.click().run(timeout=20)
    lobe_tab = next(t for t in at.tabs if t.label == "Lobe Manager")
    assert any("self-attention" in s.value.lower() for s in lobe_tab.success)

    lobe_tab.number_input[1].set_value(0.1)
    sel_btn = next(b for b in lobe_tab.button if b.label == "Select High Attention")
    at = sel_btn.click().run(timeout=20)
    lobe_tab = next(t for t in at.tabs if t.label == "Lobe Manager")
    assert lobe_tab.markdown


def test_core_tools_operations():
    at = _setup_advanced_playground()
    core_tab = next(t for t in at.tabs if t.label == "Core Tools")

    core_tab.text_input[0].input("standard")
    core_tab.text_input[1].input("")
    add_n = next(b for b in core_tab.button if b.label == "Add Neuron")
    at = add_n.click().run(timeout=20)
    core_tab = next(t for t in at.tabs if t.label == "Core Tools")
    assert any("added" in md.value.lower() for md in core_tab.markdown)

    core_tab.number_input[3].set_value(0)
    core_tab.number_input[4].set_value(1)
    core_tab.number_input[5].set_value(0.5)
    core_tab.text_input[3].input("standard")
    add_s = next(b for b in core_tab.button if b.label == "Add Synapse")
    at = add_s.click().run(timeout=20)
    core_tab = next(t for t in at.tabs if t.label == "Core Tools")
    assert any("synapse added" in s.value.lower() for s in core_tab.success)

    core_tab.number_input[6].set_value(0.1)
    freeze_btn = next(b for b in core_tab.button if b.label == "Freeze")
    at = freeze_btn.click().run(timeout=20)
    core_tab = next(t for t in at.tabs if t.label == "Core Tools")
    assert any("frozen" in s.value.lower() for s in core_tab.success)

    core_tab.number_input[7].set_value(1)
    inc_btn = next(b for b in core_tab.button if b.label == "Increase Representation")
    at = inc_btn.click().run(timeout=20)
    core_tab = next(t for t in at.tabs if t.label == "Core Tools")
    assert any("increased" in s.value.lower() for s in core_tab.success)

    core_tab.number_input[8].set_value(1)
    dec_btn = next(b for b in core_tab.button if b.label == "Decrease Representation")
    at = dec_btn.click().run(timeout=20)
    core_tab = next(t for t in at.tabs if t.label == "Core Tools")
    assert any("decreased" in s.value.lower() for s in core_tab.success)

    core_tab.number_input[9].set_value(1)
    mp_btn = next(b for b in core_tab.button if b.label == "Run Message Passing")
    at = mp_btn.click().run(timeout=20)
    core_tab = next(t for t in at.tabs if t.label == "Core Tools")
    assert any("avg change" in md.value.lower() for md in core_tab.markdown)

    reset_btn = next(b for b in core_tab.button if b.label == "Reset Representations")
    at = reset_btn.click().run(timeout=20)
    core_tab = next(t for t in at.tabs if t.label == "Core Tools")
    assert any("reset" in s.value.lower() for s in core_tab.success)

    core_tab.number_input[10].set_value(1.0)
    rand_btn = next(
        b for b in core_tab.button if b.label == "Randomize Representations"
    )
    at = rand_btn.click().run(timeout=20)
    core_tab = next(t for t in at.tabs if t.label == "Core Tools")
    assert any("randomized" in s.value.lower() for s in core_tab.success)


def test_config_editor_reinitialize():
    at = _setup_advanced_playground()
    cfg_tab = next(t for t in at.tabs if t.label == "Config Editor")

    cfg_tab.text_input[0].input("core.width")
    cfg_tab.text_input[1].input("5")
    at = cfg_tab.button[0].click().run(timeout=20)
    cfg_tab = next(t for t in at.tabs if t.label == "Config Editor")
    at = cfg_tab.button[1].click().run(timeout=20)
    cfg_tab = next(t for t in at.tabs if t.label == "Config Editor")
    assert any("reinitialized" in s.value.lower() for s in cfg_tab.success)


def test_adaptive_control_actions(monkeypatch):
    monkeypatch.setattr(
        "streamlit_playground.adjust_meta_controller", lambda *a, **k: 0.5
    )
    at = _setup_advanced_playground()
    adapt_tab = next(t for t in at.tabs if t.label == "Adaptive Control")

    adj_btn = next(b for b in adapt_tab.button if b.label == "Adjust Now")
    at = adj_btn.click().run(timeout=20)
    adapt_tab = next(t for t in at.tabs if t.label == "Adaptive Control")
    assert any("threshold" in s.value.lower() for s in adapt_tab.success)

    clr_btn = next(b for b in adapt_tab.button if b.label == "Clear History")
    at = clr_btn.click().run(timeout=20)
    adapt_tab = next(t for t in at.tabs if t.label == "Adaptive Control")
    assert any("history cleared" in s.value.lower() for s in adapt_tab.success)


def test_projects_tab_run(monkeypatch):
    monkeypatch.setattr("streamlit_playground.run_example_project", lambda p: "done")
    at = _setup_advanced_playground()
    proj_tab = next(t for t in at.tabs if t.label == "Projects")

    proj_tab.selectbox[0].set_value("project05_gpt_training.py")
    run_btn = next(b for b in proj_tab.button if b.label == "Run Project")
    at = run_btn.click().run(timeout=20)
    proj_tab = next(t for t in at.tabs if t.label == "Projects")
    assert any("done" in t.value for t in proj_tab.text)


def test_basic_inference():
    at = _setup_basic_playground()
    num_input = next(n for n in at.number_input if n.label == "Numeric Input")
    num_input.set_value(0.5)
    infer_btn = next(b for b in at.button if b.label == "Infer")
    at = infer_btn.click().run(timeout=20)
    assert any("Output:" in md.value for md in at.markdown)


def test_offload_and_convert(monkeypatch):
    flags = {"remote": False, "torrent": False, "convert": False}

    def fake_offload(self, threshold=1.0):
        flags["remote"] = True

    def fake_offload_torrent(self, threshold=1.0):
        flags["torrent"] = True

    def fake_convert(name):
        flags["convert"] = True
        return "marble"

    monkeypatch.setattr(
        "marble_brain.Brain.offload_high_attention",
        fake_offload,
    )
    monkeypatch.setattr(
        "marble_brain.Brain.offload_high_attention_torrent",
        fake_offload_torrent,
    )
    monkeypatch.setattr(
        "streamlit_playground.convert_hf_model",
        fake_convert,
    )

    at = _setup_advanced_playground()

    model_tab = next(t for t in at.tabs if t.label == "Model Conversion")
    model_tab.text_input[1].input("dummy-model")
    at = model_tab.run(timeout=20)

    off_tab = next(t for t in at.tabs if t.label == "Offloading")
    off_btn = next(b for b in off_tab.button if b.label == "Offload High Attention")
    at = off_btn.click().run(timeout=20)

    tor_btn = next(b for b in off_tab.button if b.label == "Offload via Torrent")
    at = tor_btn.click().run(timeout=20)

    conv_btn = next(b for b in off_tab.button if b.label == "Convert to MARBLE")
    at = conv_btn.click().run(timeout=20)
    off_tab = next(t for t in at.tabs if t.label == "Offloading")

    assert flags["remote"] and flags["torrent"] and flags["convert"]
    assert any("Model converted" in s.value for s in off_tab.success)


def test_sidebar_toggles_and_dashboard(monkeypatch):
    dash = type("Dash", (), {"stop": lambda self: None})()
    monkeypatch.setattr(
        "streamlit_playground.start_metrics_dashboard", lambda *a, **k: dash
    )
    monkeypatch.setattr("streamlit_playground.set_dreaming", lambda *a, **k: None)
    monkeypatch.setattr("streamlit_playground.set_autograd", lambda *a, **k: None)

    at = _setup_basic_playground()
    dream_cb = next(c for c in at.sidebar.checkbox if c.label == "Dreaming")
    auto_cb = next(c for c in at.sidebar.checkbox if c.label == "Autograd")
    dash_cb = next(c for c in at.sidebar.checkbox if c.label == "Metrics Dashboard")

    at = dream_cb.toggle().run(timeout=20)
    at = auto_cb.toggle().run(timeout=20)
    at = dash_cb.toggle().run(timeout=20)
    assert at.session_state["dashboard"] is dash

    at = dash_cb.toggle().run(timeout=20)
    assert at.session_state["dashboard"] is None


def test_training_sidebar_action(monkeypatch):
    monkeypatch.setattr(
        "streamlit_playground.train_marble_system", lambda *a, **k: None
    )
    at = _setup_basic_playground()
    at.session_state["hf_examples"] = [(1, 2)]
    train_btn = next(b for b in at.sidebar.button if b.label == "Train")
    at = train_btn.click().run(timeout=20)
    assert any("Training complete" in s.value for s in at.sidebar.success)


def test_basic_text_inference():
    at = _setup_basic_playground()
    txt_in = next(t for t in at.text_input if t.label == "Text Input")
    txt_in.input("hello")
    infer_btn = next(b for b in at.button if b.label == "Infer")
    at = infer_btn.click().run(timeout=20)
    assert any("Output:" in md.value for md in at.markdown)


def test_save_config_file(monkeypatch):
    at = _setup_basic_playground()
    exp = at.sidebar.expander[0]
    monkeypatch.setattr(
        "streamlit_playground.save_config_yaml", lambda text, path: None
    )
    exp.text_input[0].input("saved.yaml")
    at = exp.button[0].click().run(timeout=20)
    assert any("Config saved" in s.value for s in at.sidebar.success)


def test_duplicate_instance(monkeypatch):
    monkeypatch.setattr("streamlit_playground.save_marble_system", lambda *a, **k: None)
    monkeypatch.setattr(
        "streamlit_playground.MarbleRegistry.duplicate", lambda *a, **k: None
    )
    at = _setup_basic_playground()
    dup_btn = next(b for b in at.sidebar.button if b.label == "Duplicate Instance")
    at = dup_btn.click().run(timeout=20)
    assert any("duplicated" in s.value.lower() for s in at.sidebar.success)


def test_pipeline_reorder_and_remove():
    at = _setup_advanced_playground()
    pipe_tab = next(t for t in at.tabs if t.label == "Pipeline")
    add_exp = pipe_tab.expander[1]
    add_exp.selectbox[0].set_value("count_marble_synapses")
    at = add_exp.button[0].click().run(timeout=20)
    pipe_tab = next(t for t in at.tabs if t.label == "Pipeline")
    add_exp = pipe_tab.expander[1]
    add_exp.selectbox[0].set_value("count_marble_synapses")
    at = add_exp.button[0].click().run(timeout=20)
    pipe_tab = next(t for t in at.tabs if t.label == "Pipeline")
    assert len(at.session_state["pipeline"]) == 2
    down_btn = next(b for b in pipe_tab.button if b.label == "⬇")
    at = down_btn.click().run(timeout=20)
    pipe_tab = next(t for t in at.tabs if t.label == "Pipeline")
    assert len(at.session_state["pipeline"]) == 2
    rm_btn = next(b for b in pipe_tab.button if b.label == "✕")
    at = rm_btn.click().run(timeout=20)
    assert len(at.session_state["pipeline"]) == 1


def test_auto_firing_start_stop():
    at = _setup_advanced_playground()
    async_tab = next(t for t in at.tabs if t.label == "Async Training")
    start_btn = next(b for b in async_tab.button if b.label == "Start Auto-Firing")
    at = start_btn.click().run(timeout=20)
    async_tab = next(t for t in at.tabs if t.label == "Async Training")
    assert any("started" in s.value.lower() for s in async_tab.success)
    stop_btn = next(b for b in async_tab.button if b.label == "Stop Auto-Firing")
    at = stop_btn.click().run(timeout=20)
    async_tab = next(t for t in at.tabs if t.label == "Async Training")
    assert any("stopped" in s.value.lower() for s in async_tab.success)


def test_save_marble(monkeypatch, tmp_path):
    monkeypatch.setattr("streamlit_playground.save_marble_system", lambda m, p: None)
    monkeypatch.setattr("marble_interface.save_marble_system", lambda m, p: None)
    at = _setup_basic_playground()
    path_input = next(t for t in at.sidebar.text_input if t.label == "Save Path")
    path_input.input(str(tmp_path / "model.pkl"))
    save_btn = next(b for b in at.sidebar.button if b.label == "Save MARBLE")
    at = save_btn.click().run(timeout=20)
    assert any("Model saved" in s.value for s in at.sidebar.success)


def test_create_instance_button(monkeypatch):
    monkeypatch.setattr(
        "streamlit_playground.MarbleRegistry.create", lambda *a, **k: object()
    )
    at = AppTest.from_file("streamlit_playground.py").run(timeout=15)
    create_btn = next(b for b in at.sidebar.button if b.label == "Create Instance")
    at = create_btn.click().run(timeout=20)
    assert any("created" in s.value for s in at.sidebar.success)


def test_attach_remote_client(monkeypatch):
    monkeypatch.setattr(
        "streamlit_playground.create_remote_client", lambda url: object()
    )
    at = _setup_advanced_playground()
    off_tab = next(t for t in at.tabs if t.label == "Offloading")
    client_exp = off_tab.expander[1]
    at = client_exp.button[0].click().run(timeout=20)
    client_exp = next(t for t in at.tabs if t.label == "Offloading").expander[1]
    attach_btn = next(b for b in client_exp.button if b.label == "Attach to MARBLE")
    at = attach_btn.click().run(timeout=20)
    off_tab = next(t for t in at.tabs if t.label == "Offloading")
    assert any("attached" in s.value for s in off_tab.success)


def test_stats_refresh():
    at = _setup_advanced_playground()
    stats_tab = next(t for t in at.tabs if t.label == "System Stats")
    btn = next(b for b in stats_tab.button if b.label == "Refresh Stats")
    at = btn.click().run(timeout=20)
    stats_tab = next(t for t in at.tabs if t.label == "System Stats")
    assert len(stats_tab.metric) == 2


def test_misc_gui_buttons():
    at = _setup_advanced_playground()
    vis_tab = next(t for t in at.tabs if t.label == "Visualization")
    vis_tab.selectbox[0].set_value("circular")
    graph_btn = next(b for b in vis_tab.button if b.label == "Generate Graph")
    at = graph_btn.click().run(timeout=20)
    vis_tab = next(t for t in at.tabs if t.label == "Visualization")
    assert vis_tab.get("plotly_chart")
    heat_tab = next(t for t in at.tabs if t.label == "Weight Heatmap")
    heat_tab.number_input[0].set_value(5)
    heat_tab.selectbox[0].set_value("Cividis")
    heat_btn = next(b for b in heat_tab.button if b.label == "Generate Heatmap")
    at = heat_btn.click().run(timeout=20)
    heat_tab = next(t for t in at.tabs if t.label == "Weight Heatmap")
    assert heat_tab.get("plotly_chart")
    code_tab = next(t for t in at.tabs if t.label == "Custom Code")
    code_tab.text_area[0].input("x = 1")
    run_btn = next(b for b in code_tab.button if b.label == "Run Code")
    at = run_btn.click().run(timeout=20)
    code_tab = next(t for t in at.tabs if t.label == "Custom Code")
    assert code_tab.markdown


def test_start_background_training(monkeypatch):
    monkeypatch.setattr(
        "streamlit_playground.start_background_training", lambda *a, **k: None
    )
    monkeypatch.setattr("streamlit_playground.wait_for_training", lambda *a, **k: None)
    at = _setup_advanced_playground()
    at.session_state["hf_examples"] = [(1, 2)]
    async_tab = next(t for t in at.tabs if t.label == "Async Training")
    start_btn = next(
        b for b in async_tab.button if b.label == "Start Background Training"
    )
    at = start_btn.click().run(timeout=20)
    async_tab = next(t for t in at.tabs if t.label == "Async Training")
    assert any("Training started" in s.value for s in async_tab.success)
    wait_btn = next(b for b in async_tab.button if b.label == "Wait For Training")
    at = wait_btn.click().run(timeout=20)
    async_tab = next(t for t in at.tabs if t.label == "Async Training")
    assert any("complete" in s.value.lower() for s in async_tab.success)


def test_model_search_and_preview(monkeypatch):
    monkeypatch.setattr(
        "streamlit_playground.search_hf_models",
        lambda q: ["dummy-model"],
    )

    class Dummy:
        def named_parameters(self):
            return []

    monkeypatch.setattr(
        "streamlit_playground.load_hf_model",
        lambda name: Dummy(),
    )
    at = _setup_advanced_playground()
    model_tab = next(t for t in at.tabs if t.label == "Model Conversion")
    model_tab.text_input[0].input("dummy")
    at = model_tab.button[0].click().run(timeout=20)
    model_tab = next(t for t in at.tabs if t.label == "Model Conversion")
    model_tab.selectbox[0].set_value("dummy-model")
    at = model_tab.button[1].click().run(timeout=20)
    model_tab = next(t for t in at.tabs if t.label == "Model Conversion")
    assert any("Layer" in txt.value or "Total" in txt.value for txt in model_tab.text)


def test_adaptive_control_extra(monkeypatch):
    monkeypatch.setattr(
        "streamlit_playground.update_meta_controller",
        lambda *a, **k: {"history_length": 1},
    )
    monkeypatch.setattr(
        "streamlit_playground.super_evo_changes", lambda m: [{"step": 1}]
    )
    monkeypatch.setattr("streamlit_playground.clear_super_evo_changes", lambda m: None)
    monkeypatch.setattr(
        "streamlit_playground.run_dimensional_search", lambda m, loss: 1
    )
    monkeypatch.setattr("streamlit_playground.run_nd_topology", lambda m, loss: 1)

    at = _setup_advanced_playground()
    marble = at.session_state["marble"]
    marble.get_brain().dim_search = object()
    marble.get_brain().nd_topology = object()
    adapt_tab = next(t for t in at.tabs if t.label == "Adaptive Control")

    exp = adapt_tab.expander[0]
    exp.number_input[0].set_value(2)
    exp.number_input[1].set_value(0.1)
    exp.number_input[2].set_value(0.1)
    exp.number_input[3].set_value(1.0)
    at = exp.button[0].click().run(timeout=20)
    adapt_tab = next(t for t in at.tabs if t.label == "Adaptive Control")
    assert adapt_tab.json

    clr_btn = next(b for b in adapt_tab.button if b.label == "Clear Change Log")
    at = clr_btn.click().run(timeout=20)
    adapt_tab = next(t for t in at.tabs if t.label == "Adaptive Control")
    assert any("Cleared" in s.value for s in adapt_tab.success)

    ds_btn = next(
        b for b in adapt_tab.button if b.label == "Evaluate Dimensional Search"
    )
    at = ds_btn.click().run(timeout=20)
    adapt_tab = next(t for t in at.tabs if t.label == "Adaptive Control")
    nd_btn = next(b for b in adapt_tab.button if b.label == "Evaluate N-D Topology")
    at = nd_btn.click().run(timeout=20)
    adapt_tab = next(t for t in at.tabs if t.label == "Adaptive Control")
    assert any("Representation size" in md.value for md in adapt_tab.markdown)


def test_hybrid_memory_forget(monkeypatch):
    monkeypatch.setattr(
        "streamlit_playground.hybrid_memory_forget", lambda *a, **k: None
    )
    at = _setup_advanced_playground()
    hm_tab = next(t for t in at.tabs if t.label == "Hybrid Memory")
    hm_tab.text_input[0].input("vec.pkl")
    hm_tab.text_input[1].input("sym.pkl")
    at = hm_tab.button[0].click().run(timeout=20)
    hm_tab = next(t for t in at.tabs if t.label == "Hybrid Memory")
    hm_tab.number_input[2].set_value(10)
    forget_btn = next(b for b in hm_tab.button if b.label == "Forget Old")
    at = forget_btn.click().run(timeout=20)
    hm_tab = next(t for t in at.tabs if t.label == "Hybrid Memory")
    assert any("Pruned" in s.value for s in hm_tab.success)


def test_instance_switch_and_delete(monkeypatch):
    monkeypatch.setattr("streamlit_playground.save_marble_system", lambda *a, **k: None)

    def fake_dup(self, src, new):
        self.instances[new] = self.instances[src]

    def fake_del(self, name):
        self.instances.pop(name, None)

    monkeypatch.setattr("streamlit_playground.MarbleRegistry.duplicate", fake_dup)
    monkeypatch.setattr("streamlit_playground.MarbleRegistry.delete", fake_del)

    at = _setup_basic_playground()
    dup_btn = next(b for b in at.sidebar.button if b.label == "Duplicate Instance")
    at = dup_btn.click().run(timeout=20)
    at = at.run(timeout=20)
    sb_select = at.sidebar.selectbox[0]
    sb_select.set_value("main_copy")
    switch_btn = next(b for b in at.sidebar.button if b.label == "Switch Instance")
    at = switch_btn.click().run(timeout=20)
    del_btn = next(b for b in at.sidebar.button if b.label == "Delete Instance")
    at = del_btn.click().run(timeout=20)


def test_about_dialog():
    at = AppTest.from_file("streamlit_playground.py").run(timeout=15)
    about_btn = next(b for b in at.button if b.label == "About")
    at = about_btn.click().run(timeout=20)
    assert any("MARBLE" in md.value for md in at.markdown)


def test_autoencoder_tab_training(monkeypatch):
    monkeypatch.setattr("streamlit_playground.train_autoencoder", lambda *a, **k: 0.0)
    at = _setup_advanced_playground()
    auto_tab = next(t for t in at.tabs if t.label == "Autoencoder")
    auto_tab.file_uploader[0].upload(("vals.csv", "value\n0.1\n0.2\n"))
    train_btn = next(b for b in auto_tab.button if b.label == "Train Autoencoder")
    at = train_btn.click().run(timeout=20)
    auto_tab = next(t for t in at.tabs if t.label == "Autoencoder")
    assert any("Training complete" in s.value for s in auto_tab.success)


def test_persist_ui_state_inserts_script(monkeypatch):
    captured = {}

    def fake_html(data, **kwargs):
        captured["data"] = data

    monkeypatch.setattr("streamlit.components.v1.html", fake_html)
    import importlib

    import streamlit_playground

    importlib.reload(streamlit_playground)
    streamlit_playground._persist_ui_state()
    assert "scrollPos" in captured.get("data", "")


def test_dataset_browser_load(tmp_path):
    csv = tmp_path / "sample.csv"
    csv.write_text("input,target\n1,2\n3,4\n")
    at = _setup_advanced_playground()
    tab = next(t for t in at.tabs if t.label == "Dataset Browser")
    assert tab.label == "Dataset Browser"
