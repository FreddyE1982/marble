import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tests.test_core_functions import minimal_params
from marble_core import Core
from marble_neuronenblitz import Neuronenblitz
from core_interconnect import interconnect_cores
from omni_learning import OmniLearner


def test_interconnect_and_train():
    params = minimal_params()
    core1 = Core(params)
    core2 = Core(params)
    combined = interconnect_cores([core1, core2], prob=1.0)
    expected = len(core1.neurons) + len(core2.neurons)
    nb = Neuronenblitz(combined)
    learner = OmniLearner(combined, nb)
    learner.train([(0.1, 0.2)], epochs=1)
    assert len(combined.neurons) >= expected
    inter_syn = [s for s in combined.synapses if s.synapse_type == "interconnection"]
    assert inter_syn

