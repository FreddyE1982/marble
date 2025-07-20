from tests.test_core_functions import minimal_params
from marble_core import Core
from marble_neuronenblitz import Neuronenblitz
from synaptic_echo_learning import SynapticEchoLearner


def test_synaptic_echo_learning_runs():
    params = minimal_params()
    params['synapse_echo_length'] = 3
    params['synapse_echo_decay'] = 0.8
    core = Core(params)
    nb = Neuronenblitz(core, use_echo_modulation=True)
    learner = SynapticEchoLearner(core, nb)
    err = learner.train_step(0.5, 1.0)
    assert isinstance(err, float)
    assert any(s.echo_buffer for s in core.synapses)
    assert learner.history
