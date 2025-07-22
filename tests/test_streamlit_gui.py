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
