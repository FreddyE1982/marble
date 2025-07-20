import random
import numpy as np
from tests.test_core_functions import minimal_params
from marble_core import Core, Neuron
from marble_neuronenblitz import Neuronenblitz
from meta_learning import MetaLearner


def create_meta():
    random.seed(0)
    np.random.seed(0)
    params = minimal_params()
    core = Core(params)
    core.neurons = [Neuron(0, value=0.0), Neuron(1, value=0.0)]
    core.synapses = []
    core.add_synapse(0, 1, weight=0.1)
    nb = Neuronenblitz(
        core,
        split_probability=0.0,
        alternative_connection_prob=0.0,
        backtrack_probability=0.0,
        backtrack_enabled=False,
    )
    return core, nb


def test_meta_learning_updates_weights():
    core, nb = create_meta()
    learner = MetaLearner(core, nb, inner_steps=1, meta_lr=0.5)
    tasks = [[(1.0, 2.0)], [(1.0, 3.0)]]
    old_w = core.synapses[0].weight
    loss = learner.train_step(tasks)
    assert isinstance(loss, float)
    assert core.synapses[0].weight != old_w
    assert len(learner.history) == 1