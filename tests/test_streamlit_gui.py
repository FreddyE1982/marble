import os
import sys
import warnings

# Suppress protobuf deprecation warnings from dependencies before importing
warnings.filterwarnings(
    "ignore",
    message=".*PyType_Spec.*",
    category=DeprecationWarning,
)
from _pytest.warning_types import PytestDeprecationWarning
warnings.filterwarnings(
    "ignore",
    category=PytestDeprecationWarning,
    module="dash.testing.plugin",
)

from streamlit.testing.v1 import AppTest

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
    vis_tab.button[0].click()
    at = vis_tab.run(timeout=20)
    vis_tab = next(t for t in at.tabs if t.label == "Visualization")
    assert vis_tab.get("plotly_chart")

    heat_tab = next(t for t in at.tabs if t.label == "Weight Heatmap")
    heat_tab.number_input[0].set_value(10)
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
    assert (
        proj_tab.expander[0].code
        and "import" in proj_tab.expander[0].code[0].value
    )


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
    monkeypatch.setattr(
        "streamlit_playground.start_remote_server", lambda **kw: dummy
    )
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
