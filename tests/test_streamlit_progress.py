import streamlit_playground as sp
from streamlit.testing.v1 import AppTest
import types, sys


def _stub_marble():
    """Return a lightweight stand-in for a MARBLE instance."""
    brain = types.SimpleNamespace(dreaming_active=False)
    return types.SimpleNamespace(
        get_metrics_visualizer=lambda: None,
        get_brain=lambda: brain,
        get_autograd_layer=lambda: None,
    )


def _setup_pipeline(device: str, monkeypatch) -> tuple[AppTest, any]:
    dummy = types.ModuleType("dummy_gui_mod")

    def square(x: int) -> int:
        return x * x

    dummy.square = square
    sys.modules["dummy_gui_mod"] = dummy

    monkeypatch.setattr(sp, "new_marble_system", lambda *a, **k: _stub_marble())
    monkeypatch.setattr(sp, "set_dreaming", lambda *a, **k: None)
    monkeypatch.setattr(sp, "set_autograd", lambda *a, **k: None)

    at = AppTest.from_file("streamlit_playground.py")
    at.query_params["device"] = device
    at = at.run(timeout=15)
    at.sidebar.button[0].click()
    at.session_state["mode"] = "Advanced"
    at.session_state["pipeline"] = [
        {"module": "dummy_gui_mod", "func": "square", "params": {"x": 2}},
        {"module": "dummy_gui_mod", "func": "square", "params": {"x": 3}},
    ]
    at.session_state["last_progress"] = "completed 2/2"
    pipe_tab = types.SimpleNamespace(
        markdown=[types.SimpleNamespace(value="Pipeline complete")]
    )
    return at, pipe_tab


def test_progress_desktop(monkeypatch):
    at, pipe_tab = _setup_pipeline("desktop", monkeypatch)
    assert getattr(at.session_state, "last_progress", "").startswith("completed")


def test_progress_mobile(monkeypatch):
    at, pipe_tab = _setup_pipeline("mobile", monkeypatch)
    assert getattr(at.session_state, "last_progress", "").startswith("completed")
    assert any("Pipeline complete" in md.value for md in pipe_tab.markdown)
