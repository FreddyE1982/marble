import random
import numpy as np
import torch
import pytest
from marble_core import Core, Neuron
from marble_neuronenblitz import Neuronenblitz
import marble_neuronenblitz.core as nb_core
from tests.test_core_functions import minimal_params


def _simple_core():
    params = minimal_params()
    core = Core(params)
    core.neurons = [Neuron(0, value=1.0), Neuron(1, value=0.0)]
    core.add_synapse(0, 1, weight=1.0)
    return core


def test_custom_loss_fn_no_update():
    random.seed(0)
    np.random.seed(0)
    core = _simple_core()
    nb = Neuronenblitz(core, consolidation_probability=0.0, weight_decay=0.0,
                       split_probability=0.0, alternative_connection_prob=0.0,
                       backtrack_probability=0.0, backtrack_enabled=False,
                       structural_plasticity_enabled=False)
    nb.learning_rate = 1.0
    nb.decide_synapse_action = lambda: None
    nb.apply_structural_plasticity = lambda path: None
    nb_core.perform_message_passing = lambda *a, **k: 0
    core.expand = lambda *a, **k: None
    syn = core.synapses[0]
    initial = syn.weight
    nb.train([(1.0, 0.0)], epochs=1, loss_fn=lambda t, o: 0.0)
    assert syn.weight == initial


def test_validation_fn_scales_error():
    random.seed(0)
    np.random.seed(0)
    core = _simple_core()
    nb = Neuronenblitz(core, consolidation_probability=0.0, weight_decay=0.0,
                       split_probability=0.0, alternative_connection_prob=0.0,
                       backtrack_probability=0.0, backtrack_enabled=False,
                       structural_plasticity_enabled=False)
    nb.learning_rate = 1.0
    nb.decide_synapse_action = lambda: None
    nb.apply_structural_plasticity = lambda path: None
    nb_core.perform_message_passing = lambda *a, **k: 0
    core.expand = lambda *a, **k: None
    syn = core.synapses[0]
    initial = syn.weight
    nb.train([(1.0, 0.0)], epochs=1, validation_fn=lambda t, o: 0.0)
    assert syn.weight == initial


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_validation_fn_gpu():
    random.seed(0)
    np.random.seed(0)
    core = _simple_core()
    core.neurons[0].tier = "vram"
    core.neurons[1].tier = "vram"
    nb = Neuronenblitz(core, consolidation_probability=0.0, weight_decay=0.0,
                       split_probability=0.0, alternative_connection_prob=0.0,
                       backtrack_probability=0.0, backtrack_enabled=False,
                       structural_plasticity_enabled=False)
    nb.learning_rate = 1.0
    nb.decide_synapse_action = lambda: None
    nb.apply_structural_plasticity = lambda path: None
    nb_core.perform_message_passing = lambda *a, **k: 0
    core.expand = lambda *a, **k: None
    syn = core.synapses[0]
    initial = syn.weight
    nb.train([(1.0, 0.0)], epochs=1, validation_fn=lambda t, o: 0.0)
    assert syn.weight == initial
