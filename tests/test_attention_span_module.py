import torch
from marble_core import Core, Neuron
from marble_neuronenblitz import Neuronenblitz, DynamicSpanModule
from tests.test_core_functions import minimal_params


def test_span_module_selects_threshold():
    module = DynamicSpanModule(max_span=3, threshold=0.6)
    scores = torch.tensor([[0.1, 0.2, 0.3, 0.4]])
    mask = module(scores)
    assert mask.shape == scores.shape
    # cumulative softmax results keep first three elements under 0.6
    assert mask[0, 0] and mask[0, 1]
    assert not mask[0, -1]


def test_span_module_device_fallback():
    module = DynamicSpanModule()
    if torch.cuda.is_available():
        assert module.device.type == "cuda"
    else:
        assert module.device.type == "cpu"


def test_neuronenblitz_applies_span():
    params = minimal_params()
    core = Core(params)
    core.neurons = [Neuron(i, value=0.0) for i in range(3)]
    core.add_synapse(0, 1, weight=1.0)
    core.add_synapse(1, 2, weight=1.0)
    nb = Neuronenblitz(
        core,
        max_attention_span=2,
        attention_span_threshold=0.8,
        split_probability=0.0,
        alternative_connection_prob=0.0,
        backtrack_probability=0.0,
        backtrack_enabled=False,
    )
    core.neurons[0].attention_score = 1.0
    core.neurons[1].attention_score = 0.5
    core.neurons[2].attention_score = 0.1
    _, path = nb.dynamic_wander(0.5)
    assert len(path) <= 2
