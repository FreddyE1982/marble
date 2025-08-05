import random
import numpy as np
import pytest
import torch

from marble_neuronenblitz import Neuronenblitz
from tests.test_neuronenblitz_enhancements import create_simple_core


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
