import os, sys
import random
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from marble_core import Core, Neuron
from marble_neuronenblitz import Neuronenblitz
from tests.test_core_functions import minimal_params


def create_linear_core():
    params = minimal_params()
    core = Core(params)
    core.neurons = [Neuron(0, value=0.0), Neuron(1, value=0.0)]
    core.synapses = []
    syn = core.add_synapse(0, 1, weight=0.0)
    return core, syn


def test_genetic_algorithm_improves_error():
    random.seed(0)
    np.random.seed(0)
    core, syn = create_linear_core()
    nb = Neuronenblitz(core)
    data = [(1.0, 2.0), (2.0, 4.0)]
    out_before, _ = nb.dynamic_wander(1.0, apply_plasticity=False)
    nb.genetic_algorithm(
        data,
        population_size=4,
        generations=3,
        mutation_rate=0.5,
        mutation_strength=1.0,
        selection_ratio=0.5,
    )
    out_after, _ = nb.dynamic_wander(1.0, apply_plasticity=False)
    loss_before = nb._compute_loss(2.0, out_before)
    loss_after = nb._compute_loss(2.0, out_after)
    assert loss_after <= loss_before
    assert nb.core.synapses[0].weight != 0.0
