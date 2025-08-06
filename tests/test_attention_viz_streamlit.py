from streamlit.testing.v1 import AppTest
import streamlit_playground as sp


def test_attention_mask_gate_visualization():
    at = AppTest.from_file("streamlit_playground.py")
    at = at.run(timeout=15)
    at = at.sidebar.button[0].click().run(timeout=30)
    at = at.sidebar.radio[0].set_value("Advanced").run(timeout=20)
    vis_tab = next(t for t in at.tabs if t.label == "Visualization")
    exp = next(e for e in vis_tab.expander if e.label == "Attention Mask & Gating")
    exp.number_input[0].set_value(4)
    exp.selectbox[0].set_value("sine")
    at = exp.button[0].click().run(timeout=15)
    vis_tab = next(t for t in at.tabs if t.label == "Visualization")
    exp = next(e for e in vis_tab.expander if e.label == "Attention Mask & Gating")
    assert exp.get("plotly_chart")
