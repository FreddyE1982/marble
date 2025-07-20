from tests.test_core_functions import minimal_params
from marble_core import Core, Neuron
from marble_neuronenblitz import Neuronenblitz
from continual_learning import ReplayContinualLearner


def create_setup():
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


def test_continual_learning_train():
    core, nb = create_setup()
    learner = ReplayContinualLearner(core, nb, memory_size=2)
    learner.train([(0.5, 1.0)], epochs=1)
    assert len(learner.history) > 0
