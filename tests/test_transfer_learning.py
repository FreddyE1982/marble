from tests.test_core_functions import minimal_params
from marble_core import Core
from marble_neuronenblitz import Neuronenblitz
from transfer_learning import TransferLearner


def test_freeze_fraction_of_synapses():
    params = minimal_params()
    core = Core(params)
    nb = Neuronenblitz(core)
    core.expand(num_new_neurons=2, num_new_synapses=2)
    core.freeze_fraction_of_synapses(0.5)
    frozen = [s for s in core.synapses if getattr(s, "frozen", False)]
    assert 0 < len(frozen) <= len(core.synapses)


def test_transfer_learning_runs():
    params = minimal_params()
    core = Core(params)
    nb = Neuronenblitz(core)
    learner = TransferLearner(core, nb, freeze_fraction=0.5)
    loss = learner.train_step(0.5, 1.0)
    assert isinstance(loss, float)
    assert learner.history
