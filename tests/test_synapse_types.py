import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from marble_core import Core, SYNAPSE_TYPES
from marble_neuronenblitz import Neuronenblitz
from tests.test_core_functions import minimal_params


def test_add_synapse_with_type():
    params = minimal_params()
    core = Core(params)
    syn = core.add_synapse(0, 1, weight=1.0, synapse_type="mirror")
    assert syn.synapse_type == "mirror"
    assert syn in core.synapses
    assert syn in core.neurons[0].synapses


def test_decide_synapse_action_creates_or_removes():
    params = minimal_params()
    core = Core(params)
    nb = Neuronenblitz(core)
    nb.synapse_loss_attention["mirror"] = 1.0
    nb.synapse_size_attention["standard"] = 1.0
    initial_count = len(core.synapses)
    nb.decide_synapse_action()
    assert len(core.synapses) != initial_count
