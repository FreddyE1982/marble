import random
import numpy as np
from tests.test_core_functions import minimal_params
from marble_core import Core, Neuron
from marble_neuronenblitz import Neuronenblitz
from hebbian_learning import HebbianLearner


def create_simple_core():
    params = minimal_params()
    core = Core(params)
    core.neurons = [Neuron(0, value=0.0), Neuron(1, value=0.0)]
    core.synapses = []
    syn = core.add_synapse(0, 1, weight=0.5)
    return core, syn


def test_hebbian_learning_weight_update():
    random.seed(0)
    np.random.seed(0)
    core, syn = create_simple_core()
    nb = Neuronenblitz(
        core,
        split_probability=0.0,
        alternative_connection_prob=0.0,
        backtrack_probability=0.0,
        backtrack_enabled=False,
    )
    learner = HebbianLearner(core, nb, learning_rate=0.1)
    before = syn.weight
    learner.train_step(1.0)
    after = syn.weight
    assert after != before
