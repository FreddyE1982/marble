import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tests.test_core_functions import minimal_params
from marble_core import Core
from core_interconnect import interconnect_cores


def test_interconnection_prob_from_core_params():
    params = minimal_params()
    params["interconnection_prob"] = 1.0
    core1 = Core(params)
    core2 = Core(params)
    combined = interconnect_cores([core1, core2])
    inter_syn = [s for s in combined.synapses if s.synapse_type == "interconnection"]
    assert len(inter_syn) == len(core1.neurons) * len(core2.neurons)


def test_interconnection_prob_zero_disables_links():
    params = minimal_params()
    params["interconnection_prob"] = 0.0
    core1 = Core(params)
    core2 = Core(params)
    combined = interconnect_cores([core1, core2])
    inter_syn = [s for s in combined.synapses if s.synapse_type == "interconnection"]
    assert len(inter_syn) == 0
