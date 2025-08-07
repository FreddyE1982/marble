import streamlit_playground as sp
from streamlit.testing.v1 import AppTest
import types, sys


def _stub_marble():
    """Return a lightweight stand-in for a MARBLE instance."""
    return types.SimpleNamespace(get_metrics_visualizer=lambda: None)


def _setup_pipeline(device: str, monkeypatch) -> tuple[AppTest, any]:
    dummy = types.ModuleType("dummy_gui_mod")

    def square(x: int) -> int:
        return x * x

    dummy.square = square
    sys.modules["dummy_gui_mod"] = dummy

    monkeypatch.setattr(sp, "new_marble_system", lambda *a, **k: _stub_marble())

    at = AppTest.from_file("streamlit_playground.py")
    at.query_params["device"] = device
    at = at.run(timeout=15)
    at = at.sidebar.button[0].click().run(timeout=30)
    at = at.sidebar.radio[0].set_value("Advanced").run(timeout=20)
    pipe_tab = next(t for t in at.tabs if t.label == "Pipeline")
    at.session_state["pipeline"] = [
        {"module": "dummy_gui_mod", "func": "square", "params": {"x": 2}},
        {"module": "dummy_gui_mod", "func": "square", "params": {"x": 3}},
    ]
    run_btn = next(b for b in pipe_tab.button if b.label == "Run Pipeline")
    at = run_btn.click().run(timeout=20)
    pipe_tab = next(t for t in at.tabs if t.label == "Pipeline")
    return at, pipe_tab


def test_progress_desktop(monkeypatch):
    at, pipe_tab = _setup_pipeline("desktop", monkeypatch)
    assert getattr(at.session_state, "last_progress", "").startswith("completed")


def test_progress_mobile(monkeypatch):
    at, pipe_tab = _setup_pipeline("mobile", monkeypatch)
    assert getattr(at.session_state, "last_progress", "").startswith("completed")
    assert any("Pipeline complete" in md.value for md in pipe_tab.markdown)
