import networkx as nx
import torch
from plotly.graph_objs import Figure
from streamlit.testing.v1 import AppTest

from neural_pathway import find_neural_pathway, pathway_figure
from networkx_interop import networkx_to_core
from tests.test_core_functions import minimal_params


def test_find_neural_pathway_cpu_gpu():
    g = nx.DiGraph()
    g.add_edge(0, 1)
    g.add_edge(1, 2)
    core = networkx_to_core(g, minimal_params())
    device = "cuda" if torch.cuda.is_available() else "cpu"
    path = find_neural_pathway(core, 0, 2, device=device)
    assert path == [0, 1, 2]


def test_pathway_figure_type():
    g = nx.DiGraph()
    g.add_edge(0, 1)
    core = networkx_to_core(g, minimal_params())
    fig = pathway_figure(core, [0, 1])
    assert isinstance(fig, Figure)


def _setup_playground(timeout: float = 20) -> AppTest:
    at = AppTest.from_file("streamlit_playground.py").run(timeout=15)
    at = at.sidebar.button[0].click().run(timeout=30)
    return at.sidebar.radio[0].set_value("Advanced").run(timeout=timeout)


def test_pathway_gui_display():
    at = _setup_playground()
    nb_tab = next(t for t in at.tabs if t.label == "NB Explorer")
    exp = next(e for e in nb_tab.expander if e.label == "Neural Pathway Inspector")
    exp.number_input[0].set_value(0)
    exp.number_input[1].set_value(0)
    at = exp.button[0].click().run(timeout=20)
    nb_tab = next(t for t in at.tabs if t.label == "NB Explorer")
    exp = next(e for e in nb_tab.expander if e.label == "Neural Pathway Inspector")
    assert exp.get("plotly_chart")
