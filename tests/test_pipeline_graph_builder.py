import os
import sys
import networkx as nx

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pipeline import Pipeline
from networkx_interop import pipeline_to_networkx, pipeline_to_core
from tests.test_core_functions import minimal_params


def dummy_step(*, device: str = "cpu"):
    return device


def test_pipeline_to_graph_and_core():
    pipe = Pipeline()
    pipe.add_step("dummy_step", module=__name__, name="s1")
    pipe.add_step("dummy_step", module=__name__, name="s2", depends_on=["s1"])
    pipe.add_branch(
        branches=[
            [{"func": "dummy_step", "module": __name__, "name": "b1"}],
            [{"func": "dummy_step", "module": __name__, "name": "b2"}],
        ],
        merge={"func": "dummy_step", "module": __name__, "name": "m"},
        name="branch",
        depends_on=["s2"],
    )
    g = pipeline_to_networkx(pipe.steps)
    assert nx.is_directed_acyclic_graph(g)
    assert ("s1", "s2") in g.edges
    assert ("s2", "branch") in g.edges
    assert ("branch", "branch::b0::b1") in g.edges
    assert ("branch", "branch::b1::b2") in g.edges
    assert ("branch::b0::b1", "branch::m") in g.edges
    assert ("branch::b1::b2", "branch::m") in g.edges
    core = pipeline_to_core(pipe.steps, minimal_params())
    assert len(core.neurons) == len(g.nodes)
