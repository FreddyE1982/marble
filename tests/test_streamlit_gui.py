import os
import sys
import warnings

# Suppress protobuf deprecation warnings from dependencies before importing
warnings.filterwarnings(
    "ignore",
    message=".*PyType_Spec.*",
    category=DeprecationWarning,
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
