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


def test_excitatory_inhibitory_modulatory():
    params = minimal_params()
    core = Core(params)
    exc = core.add_synapse(0, 1, weight=-0.5, synapse_type="excitatory")
    inh = core.add_synapse(1, 2, weight=0.5, synapse_type="inhibitory")
    mod = core.add_synapse(2, 3, weight=1.0, synapse_type="modulatory")
    val1 = exc.transmit(2.0, core=core, context={})
    assert val1 > 0
    val2 = inh.transmit(val1, core=core, context={})
    assert val2 < 0
    val3 = mod.transmit(1.0, core=core, context={"reward": 1.0})
    assert val3 > 1.0


def test_mirror_side_effect():
    params = minimal_params()
    core = Core(params)
    syn = core.add_synapse(0, 1, weight=1.0, synapse_type="mirror")
    core.neurons[0].value = 0.0
    syn.transmit(2.0, core=core, context={})
    assert core.neurons[0].value == 2.0
