import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import networkx as nx
import pytest

from marble_core import Core
from networkx_interop import (
    core_to_networkx,
    networkx_to_core,
    core_to_dict,
    dict_to_core,
)
from tests.test_core_functions import minimal_params


def test_core_to_networkx_and_back():
    params = minimal_params()
    core = Core(params)
    g = core_to_networkx(core)
    assert isinstance(g, nx.DiGraph)
    new_core = networkx_to_core(g, params)
    assert len(new_core.neurons) == len(core.neurons)
    assert len(new_core.synapses) == len(core.synapses)


def test_core_dict_roundtrip():
    params = minimal_params()
    core = Core(params, formula="0", formula_num_neurons=0)

    # create two neurons and one synapse
    n0 = core.neuron_pool.allocate()
    n0.__init__(0, value=0.1, tier="vram", rep_size=core.rep_size)
    core.neurons.append(n0)
    n1 = core.neuron_pool.allocate()
    n1.__init__(1, value=0.2, tier="vram", rep_size=core.rep_size)
    core.neurons.append(n1)
    syn = core.synapse_pool.allocate()
    syn.__init__(0, 1, weight=0.3)
    core.synapses.append(syn)
    n0.synapses.append(syn)

    data = core_to_dict(core)
    assert data["nodes"][0]["id"] == 0
    assert data["edges"][0]["source"] == 0

    rebuilt = dict_to_core(data, params)
    assert len(rebuilt.neurons) == 2
    assert len(rebuilt.synapses) == 1
    assert rebuilt.synapses[0].weight == pytest.approx(0.3)
