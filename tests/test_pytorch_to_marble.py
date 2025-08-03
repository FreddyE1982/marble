import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import logging

import pytest
import torch

from marble_core import Core, Neuron
from pytorch_to_marble import (
    LAYER_CONVERTERS,
    GlobalAvgPool2d,
    TracingFailedError,
    UnsupportedLayerError,
    _add_fully_connected_layer,
    convert_model,
    register_converter,
)
from tests.test_core_functions import minimal_params


class SimpleModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.seq = torch.nn.Sequential(
            torch.nn.Linear(4, 3),
            torch.nn.ReLU(),
            torch.nn.Linear(3, 2),
        )
        self.input_size = 4

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.seq(x)


class ConvModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 1, kernel_size=2)
        self.input_size = (1, 3, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class BNModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.seq = torch.nn.Sequential(
            torch.nn.Linear(3, 3),
            torch.nn.BatchNorm1d(3, momentum=0.2),
        )
        self.input_size = 3

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.seq(x)


class LayerNormModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.seq = torch.nn.Sequential(
            torch.nn.Linear(4, 4),
            torch.nn.LayerNorm(4, eps=1e-3),
        )
        self.input_size = 4

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.seq(x)


class GroupNormModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.seq = torch.nn.Sequential(
            torch.nn.Conv2d(4, 4, kernel_size=1),
            torch.nn.GroupNorm(2, 4, eps=1e-3),
        )
        self.input_size = (4, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.seq(x)


class DropoutModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.seq = torch.nn.Sequential(
            torch.nn.Linear(2, 2),
            torch.nn.Dropout(p=0.6),
        )
        self.input_size = 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.seq(x)


class ActivationModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.seq = torch.nn.Sequential(
            torch.nn.Linear(2, 2),
            torch.nn.Sigmoid(),
        )
        self.input_size = 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.seq(x)


class TanhModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.seq = torch.nn.Sequential(
            torch.nn.Linear(2, 2),
            torch.nn.Tanh(),
        )
        self.input_size = 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.seq(x)


class FuncReluModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = torch.nn.Linear(2, 2)
        self.input_size = 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.relu(self.fc(x))


class FuncSigmoidModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = torch.nn.Linear(2, 2)
        self.input_size = 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.sigmoid(self.fc(x))


class FuncTanhModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = torch.nn.Linear(2, 2)
        self.input_size = 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.tanh(self.fc(x))


class FuncReshapeModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = torch.nn.Linear(4, 4)
        self.input_size = 4

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc(x)
        return torch.reshape(x, (2, 2))


class ViewModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = torch.nn.Linear(4, 4)
        self.input_size = 4

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc(x)
        return x.view(2, 2)


class FlattenModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.seq = torch.nn.Sequential(
            torch.nn.Conv2d(1, 1, kernel_size=2),
            torch.nn.Flatten(),
        )
        self.input_size = (1, 3, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.seq(x)


class UnflattenModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.seq = torch.nn.Sequential(
            torch.nn.Unflatten(1, (2, 2)),
        )
        self.input_size = 4

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.seq(x)


class MaxPoolModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.pool = torch.nn.MaxPool2d(2)
        self.input_size = (1, 4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(x)


class AvgPoolModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.pool = torch.nn.AvgPool2d(2)
        self.input_size = (1, 4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(x)


class GlobalAvgPoolModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.pool = GlobalAvgPool2d()
        self.input_size = (1, 4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(x)


class AdaptiveAvgPoolModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.pool = torch.nn.AdaptiveAvgPool2d((2, 2))
        self.input_size = (1, 4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(x)


class AdaptiveMaxPoolModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.pool = torch.nn.AdaptiveMaxPool2d((2, 2))
        self.input_size = (1, 4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(x)


def test_basic_conversion():
    model = SimpleModel()
    params = minimal_params()
    core = convert_model(model, core_params=params)
    assert isinstance(core, Core)
    # expect number of neurons equals input+hidden+output+biases
    expected_neurons = 4 + 3 + 1 + 2 + 1
    assert len(core.neurons) == expected_neurons
    # verify a synapse weight from first layer
    w = model.seq[0].weight.detach().cpu().numpy()[0, 0]
    syn = core.synapses[0]
    assert syn.weight == float(w)


def test_unsupported_layer():
    model = torch.nn.Sequential(torch.nn.MaxPool3d(2))
    model.input_size = (1, 1, 4, 4, 4)
    params = minimal_params()
    with pytest.raises(UnsupportedLayerError) as exc:
        convert_model(model, core_params=params)
    assert str(exc.value) == "MaxPool3d is not supported for conversion"


def test_conv2d_conversion():
    model = ConvModel()
    params = minimal_params()
    core = convert_model(model, core_params=params)
    assert any(n.neuron_type == "conv2d" for n in core.neurons)


class MultiChannelConv(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 2, kernel_size=2)
        self.input_size = (3, 3, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


def test_multichannel_conv2d_conversion():
    model = MultiChannelConv()
    params = minimal_params()
    core = convert_model(model, core_params=params)
    conv_neurons = [n for n in core.neurons if n.neuron_type == "conv2d"]
    assert len(conv_neurons) == 2


def test_batchnorm_conversion():
    model = BNModel()
    params = minimal_params()
    core = convert_model(model, core_params=params)
    assert any(n.neuron_type == "batchnorm" for n in core.neurons)
    bn_neuron = next(n for n in core.neurons if n.neuron_type == "batchnorm")
    assert bn_neuron.params["momentum"] == 0.2


def test_layernorm_conversion():
    model = LayerNormModel()
    params = minimal_params()
    core = convert_model(model, core_params=params)
    assert any(n.neuron_type == "layernorm" for n in core.neurons)
    ln_neuron = next(n for n in core.neurons if n.neuron_type == "layernorm")
    assert tuple(ln_neuron.params["normalized_shape"]) == (4,)
    assert ln_neuron.params["eps"] == 1e-3


def test_groupnorm_conversion():
    model = GroupNormModel()
    params = minimal_params()
    core = convert_model(model, core_params=params)
    assert any(n.neuron_type == "groupnorm" for n in core.neurons)
    gn_neuron = next(n for n in core.neurons if n.neuron_type == "groupnorm")
    assert gn_neuron.params["num_groups"] == 2
    assert gn_neuron.params["eps"] == 1e-3


def test_dropout_conversion():
    model = DropoutModel()
    params = minimal_params()
    core = convert_model(model, core_params=params)
    assert any(n.neuron_type == "dropout" for n in core.neurons)
    d_neuron = next(n for n in core.neurons if n.neuron_type == "dropout")
    assert d_neuron.params["p"] == 0.6


def test_activation_conversion():
    model = ActivationModel()
    params = minimal_params()
    core = convert_model(model, core_params=params)
    assert any(n.neuron_type == "sigmoid" for n in core.neurons)


def test_tanh_conversion():
    model = TanhModel()
    params = minimal_params()
    core = convert_model(model, core_params=params)
    assert any(n.neuron_type == "tanh" for n in core.neurons)


def test_flatten_conversion():
    model = FlattenModel()
    params = minimal_params()
    core = convert_model(model, core_params=params)
    assert any(n.neuron_type == "flatten" for n in core.neurons)


def test_unflatten_conversion():
    model = UnflattenModel()
    params = minimal_params()
    core = convert_model(model, core_params=params)
    assert any(n.neuron_type == "unflatten" for n in core.neurons)
    n = next(n for n in core.neurons if n.neuron_type == "unflatten")
    assert n.params["dim"] == 1
    assert tuple(n.params["unflattened_size"]) == (2, 2)


def test_maxpool2d_conversion():
    model = MaxPoolModel()
    params = minimal_params()
    core = convert_model(model, core_params=params)
    assert any(n.neuron_type == "maxpool2d" for n in core.neurons)


def test_avgpool2d_conversion():
    model = AvgPoolModel()
    params = minimal_params()
    core = convert_model(model, core_params=params)
    assert any(n.neuron_type == "avgpool2d" for n in core.neurons)


def test_globalavgpool2d_conversion():
    model = GlobalAvgPoolModel()
    params = minimal_params()
    core = convert_model(model, core_params=params)
    assert any(n.neuron_type == "avgpool2d" for n in core.neurons)


def test_adaptiveavgpool2d_conversion():
    model = AdaptiveAvgPoolModel()
    params = minimal_params()
    core = convert_model(model, core_params=params)
    assert any(n.neuron_type == "avgpool2d" for n in core.neurons)


def test_adaptivemaxpool2d_conversion():
    model = AdaptiveMaxPoolModel()
    params = minimal_params()
    core = convert_model(model, core_params=params)
    assert any(n.neuron_type == "maxpool2d" for n in core.neurons)


def test_dry_run_summary(capsys):
    model = SimpleModel()
    params = minimal_params()
    convert_model(model, core_params=params, dry_run=True)
    out = capsys.readouterr().out
    assert "created" in out
    assert "seq" in out


def test_functional_relu_conversion():
    model = FuncReluModel()
    params = minimal_params()
    core = convert_model(model, core_params=params)
    assert any(n.params.get("activation") == "relu" for n in core.neurons)


def test_functional_sigmoid_conversion():
    model = FuncSigmoidModel()
    params = minimal_params()
    core = convert_model(model, core_params=params)
    assert any(n.neuron_type == "sigmoid" for n in core.neurons)


def test_functional_tanh_conversion():
    model = FuncTanhModel()
    params = minimal_params()
    core = convert_model(model, core_params=params)
    assert any(n.neuron_type == "tanh" for n in core.neurons)


def test_functional_reshape_conversion():
    model = FuncReshapeModel()
    params = minimal_params()
    core = convert_model(model, core_params=params)
    reshape_neuron = next(n for n in core.neurons if n.neuron_type == "reshape")
    assert tuple(reshape_neuron.params["shape"]) == (2, 2)


def test_view_method_conversion():
    model = ViewModel()
    params = minimal_params()
    core = convert_model(model, core_params=params)
    assert any(n.neuron_type == "reshape" for n in core.neurons)


class FuncUnsupportedModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.input_size = (1, 3, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.max_pool2d(x, kernel_size=2)


def test_functional_unsupported_error():
    model = FuncUnsupportedModel()
    params = minimal_params()
    with pytest.raises(UnsupportedLayerError) as exc:
        convert_model(model, core_params=params)
    assert str(exc.value) == "max_pool2d is not supported for conversion"


class FailingModel(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.sum() > 0:
            return x * 2
        return x - 1


def test_tracing_failed_error():
    model = FailingModel()
    params = minimal_params()
    with pytest.raises(TracingFailedError):
        convert_model(model, core_params=params)


class DoubleLinear(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
        self.input_size = 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


@register_converter(DoubleLinear)
def _convert_doublelinear(layer: DoubleLinear, core: Core, inputs):
    out = _add_fully_connected_layer(core, inputs, layer.linear)
    for nid in out:
        core.neurons[nid].params["scale"] = 2.0
    return out


def test_custom_layer_converter():
    class Wrapper(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.custom = DoubleLinear()
            self.input_size = 2

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.custom(x)

    model = Wrapper()
    params = minimal_params()
    core = convert_model(model, core_params=params)
    assert any(n.params.get("scale") == 2.0 for n in core.neurons)
    LAYER_CONVERTERS.pop(DoubleLinear)


def test_logging_messages(caplog):
    model = SimpleModel()
    params = minimal_params()
    with caplog.at_level(logging.INFO):
        convert_model(model, core_params=params)
    assert any("Converting layer" in rec.message for rec in caplog.records)


class SequentialContainerModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.seq = torch.nn.Sequential(
            torch.nn.Linear(2, 2),
            torch.nn.ReLU(),
            torch.nn.Linear(2, 1),
        )
        self.input_size = 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.seq(x)


def test_sequential_container_conversion():
    model = SequentialContainerModel()
    params = minimal_params()
    core = convert_model(model, core_params=params)
    assert any(n.params.get("activation") == "relu" for n in core.neurons)


class ModuleListModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = torch.nn.ModuleList(
            [
                torch.nn.Linear(2, 3),
                torch.nn.Tanh(),
                torch.nn.Linear(3, 1),
            ]
        )
        self.input_size = 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


def test_modulelist_conversion():
    model = ModuleListModel()
    params = minimal_params()
    core = convert_model(model, core_params=params)
    assert any(n.neuron_type == "tanh" for n in core.neurons)


class EmbeddingModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.emb = torch.nn.Embedding(10, 3)
        self.input_size = 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.emb(x)


class EmbeddingBagModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.emb = torch.nn.EmbeddingBag(10, 3, mode="mean")
        self.input_size = 1

    def forward(
        self, x: torch.Tensor, offsets: torch.Tensor | None = None
    ) -> torch.Tensor:
        if offsets is None:
            offsets = torch.zeros(1, dtype=torch.long)
        return self.emb(x, offsets)


class EmbeddingOptionsModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.emb = torch.nn.Embedding(8, 4, padding_idx=0, max_norm=1.0)
        self.input_size = 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.emb(x)


def test_embedding_conversion():
    model = EmbeddingModel()
    params = minimal_params()
    core = convert_model(model, core_params=params)
    emb_neuron = next(n for n in core.neurons if n.neuron_type == "embedding")
    assert emb_neuron.params["num_embeddings"] == 10
    assert emb_neuron.params["embedding_dim"] == 3


class RNNModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.rnn = torch.nn.RNN(3, 2, batch_first=True)
        self.input_size = (1, 1, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.rnn(x)[0]


class LSTMModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.rnn = torch.nn.LSTM(3, 2, batch_first=True)
        self.input_size = (1, 1, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.rnn(x)[0]


class GRUModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.rnn = torch.nn.GRU(3, 2, batch_first=True)
        self.input_size = (1, 1, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.rnn(x)[0]


def test_rnn_conversion():
    model = RNNModel()
    params = minimal_params()
    core = convert_model(model, core_params=params)
    assert any(n.neuron_type == "rnn" for n in core.neurons)


def test_lstm_conversion():
    model = LSTMModel()
    params = minimal_params()
    core = convert_model(model, core_params=params)
    assert any(n.neuron_type == "lstm" for n in core.neurons)


def test_gru_conversion():
    model = GRUModel()
    params = minimal_params()
    core = convert_model(model, core_params=params)
    assert any(n.neuron_type == "gru" for n in core.neurons)


def test_embeddingbag_conversion():
    model = EmbeddingBagModel()
    params = minimal_params()
    core = convert_model(model, core_params=params)
    emb_neuron = next(n for n in core.neurons if n.neuron_type == "embedding")
    assert emb_neuron.params["num_embeddings"] == 10
    assert emb_neuron.params["embedding_dim"] == 3


def test_embedding_options_conversion():
    model = EmbeddingOptionsModel()
    params = minimal_params()
    core = convert_model(model, core_params=params)
    emb_neuron = next(n for n in core.neurons if n.neuron_type == "embedding")
    assert emb_neuron.params["padding_idx"] == 0
    assert emb_neuron.params["max_norm"] == 1.0


def test_embedding_options_conversion_gpu():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    model = EmbeddingOptionsModel().cuda()
    params = minimal_params()
    core = convert_model(model, core_params=params)
    emb_neuron = next(n for n in core.neurons if n.neuron_type == "embedding")
    assert emb_neuron.params["padding_idx"] == 0
    assert emb_neuron.params["max_norm"] == 1.0


def test_bias_synapses_values():
    model = SimpleModel()
    params = minimal_params()
    core = convert_model(model, core_params=params)
    # find bias neurons (value 1.0) created by add_fully_connected_layer
    bias_neuron_indices = [i for i, n in enumerate(core.neurons) if n.value == 1.0]
    assert len(bias_neuron_indices) == 2
    # check weights from bias neurons match layer biases
    first_bias_weight = model.seq[0].bias.detach().cpu().numpy()[0]
    syn = next(s for s in core.synapses if s.source == bias_neuron_indices[0])
    assert syn.weight == float(first_bias_weight)


def test_conversion_gpu_weights():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    model = SimpleModel().cuda()
    params = minimal_params()
    core = convert_model(model, core_params=params)
    assert len(core.neurons) > 0


def test_weight_device_info():
    layer = torch.nn.Linear(2, 2)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        layer = layer.to("cuda")
    core = Core()
    for i in range(2):
        core.neurons.append(Neuron(i, value=0.0, tier="vram"))
    out_ids = _add_fully_connected_layer(core, [0, 1], layer)
    for nid in out_ids:
        assert core.neurons[nid].params["weight_device"] == device
        if layer.bias is not None:
            assert core.neurons[nid].params["bias_device"] == device
