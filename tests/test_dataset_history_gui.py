from streamlit.testing.v1 import AppTest


def _setup_advanced_playground(timeout: float = 40) -> AppTest:
    at = AppTest.from_file("streamlit_playground.py").run(timeout=30)
    at = at.sidebar.button[0].click().run(timeout=60)
    return at.sidebar.radio[0].set_value("Advanced").run(timeout=timeout)


def test_dataset_history_controls_visible():
    at = _setup_advanced_playground()
    browser_tab = next(t for t in at.tabs if t.label == "Dataset Browser")
    assert any(e.label == "Dataset History" for e in browser_tab.expander)
