import random
import numpy as np
import pytest
import torch

from marble_core import Core, Neuron
from marble_neuronenblitz import Neuronenblitz
from tests.test_core_functions import minimal_params
from tests.test_neuronenblitz_enhancements import create_simple_core


def create_chain_core(length: int = 9):
    params = minimal_params()
    core = Core(params)
    core.neurons = [Neuron(i, value=0.0) for i in range(length + 1)]
    core.synapses = []
    syns = []
    for i in range(length):
        syns.append(core.add_synapse(i, i + 1, weight=1.0))
    return core, syns


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda_weight_update_matches_cpu():
    random.seed(0)
    np.random.seed(0)
    core_cpu, syn_cpu = create_simple_core()
    core_cpu.gradient_clip_value = 0.1
    nb_cpu = Neuronenblitz(core_cpu, consolidation_probability=0.0, weight_decay=0.0)
    nb_cpu.learning_rate = 1.0
    core_cpu.neurons[0].value = 1.0
    nb_cpu.apply_weight_updates_and_attention([syn_cpu], error=10.0)
    expected_weight = syn_cpu.weight
    expected_attention = core_cpu.neurons[1].attention_score

    core_gpu, syn_gpu = create_simple_core()
    core_gpu.gradient_clip_value = 0.1
    nb_gpu = Neuronenblitz(core_gpu, consolidation_probability=0.0, weight_decay=0.0)
    nb_gpu.learning_rate = 1.0
    core_gpu.neurons[0].value = 1.0
    nb_gpu.apply_weight_updates_and_attention([syn_gpu], error=10.0)

    assert np.isclose(syn_gpu.weight, expected_weight, atol=1e-6)
    assert np.isclose(core_gpu.neurons[1].attention_score, expected_attention, atol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda_weight_update_matches_cpu_batch():
    random.seed(0)
    np.random.seed(0)
    core_cpu, syns_cpu = create_chain_core(9)
    nb_cpu = Neuronenblitz(core_cpu, consolidation_probability=0.0, weight_decay=0.0)
    nb_cpu.learning_rate = 0.1
    for n in core_cpu.neurons[:-1]:
        n.value = 1.0
    nb_cpu.apply_weight_updates_and_attention(syns_cpu, error=1.0)
    expected_weights = [syn.weight for syn in syns_cpu]
    expected_attention = [core_cpu.neurons[i + 1].attention_score for i in range(len(syns_cpu))]

    core_gpu, syns_gpu = create_chain_core(9)
    nb_gpu = Neuronenblitz(core_gpu, consolidation_probability=0.0, weight_decay=0.0)
    nb_gpu.learning_rate = 0.1
    for n in core_gpu.neurons[:-1]:
        n.value = 1.0
    nb_gpu.apply_weight_updates_and_attention(syns_gpu, error=1.0)

    for syn, ew in zip(syns_gpu, expected_weights):
        assert np.isclose(syn.weight, ew, atol=1e-6)
    for idx, ea in enumerate(expected_attention):
        assert np.isclose(core_gpu.neurons[idx + 1].attention_score, ea, atol=1e-6)
