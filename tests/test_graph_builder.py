import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from marble_core import Core
from marble_graph_builder import (
    add_neuron_group,
    add_fully_connected_layer,
    linear_layer,
    conv2d_layer,
)
from tests.test_core_functions import minimal_params


def test_add_neuron_group_activation():
    core = Core(minimal_params(), formula="0", formula_num_neurons=0)
    ids = add_neuron_group(core, 3, activation="relu")
    assert len(ids) == 3
    for nid in ids:
        assert core.neurons[nid].params["activation"] == "relu"


def test_add_fully_connected_layer_bias_weights():
    core = Core(minimal_params(), formula="0", formula_num_neurons=0)
    inputs = add_neuron_group(core, 2)
    weights = [[1.0, 2.0], [3.0, 4.0]]
    bias = [0.5, -0.5]
    outs = add_fully_connected_layer(core, inputs, 2, weights=weights, bias=bias)
    assert len(outs) == 2
    # Check synapse weights
    assert core.synapses[0].weight == 1.0
    assert core.synapses[1].weight == 2.0
    # Bias synapses last two
    assert core.synapses[-2].weight == 0.5
    assert core.synapses[-1].weight == -0.5


def test_linear_layer_wrapper():
    core = Core(minimal_params(), formula="0", formula_num_neurons=0)
    in_ids, out_ids = linear_layer(
        core,
        2,
        2,
        weights=[[1.0, -1.0], [0.5, 0.5]],
        bias=[0.1, -0.1],
        activation="relu",
    )
    assert len(in_ids) == 2
    assert len(out_ids) == 2
    assert any(n.params.get("activation") == "relu" for n in core.neurons)
    # bias synapses should exist
    assert any(s.weight == 0.1 for s in core.synapses)


def test_conv2d_layer_wrapper():
    core = Core(minimal_params(), formula="0", formula_num_neurons=0)
    in_ids, out_ids = conv2d_layer(core, 1, 1, 2, bias=[0.2])
    assert len(in_ids) == 1
    assert len(out_ids) == 1
    conv_neuron = core.neurons[out_ids[0]]
    assert conv_neuron.neuron_type == "conv2d"
    assert "kernel" in conv_neuron.params
