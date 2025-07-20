import random
import numpy as np
from tests.test_core_functions import minimal_params
from marble_core import Core, Neuron
from marble_neuronenblitz import Neuronenblitz
from federated_learning import FederatedAveragingTrainer


def create_client(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    params = minimal_params()
    core = Core(params)
    core.neurons = [Neuron(0, value=0.0), Neuron(1, value=0.0)]
    core.synapses = []
    core.add_synapse(0, 1, weight=0.5)
    nb = Neuronenblitz(
        core,
        split_probability=0.0,
        alternative_connection_prob=0.0,
        backtrack_probability=0.0,
        backtrack_enabled=False,
    )
    return core, nb


def test_federated_round_updates_all_clients():
    c1 = create_client(0)
    c2 = create_client(1)
    trainer = FederatedAveragingTrainer([c1, c2])
    datasets = [[(1.0, 2.0)], [(0.5, 1.0)]]
    trainer.train_round(datasets, epochs=1)
    w1 = c1[0].synapses[0].weight
    w2 = c2[0].synapses[0].weight
    assert abs(w1 - w2) < 1e-6
    assert w1 != 0.5
