import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import networkx as nx

from marble_core import Core
from networkx_interop import core_to_networkx, networkx_to_core
from tests.test_core_functions import minimal_params


def test_core_to_networkx_and_back():
    params = minimal_params()
    core = Core(params)
    g = core_to_networkx(core)
    assert isinstance(g, nx.DiGraph)
    new_core = networkx_to_core(g, params)
    assert len(new_core.neurons) == len(core.neurons)
    assert len(new_core.synapses) == len(core.synapses)
