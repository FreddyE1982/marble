import json
import logging

import pytest
import torch
from torch import nn

from marble_utils import core_from_json, core_to_json, restore_hidden_states
from pytorch_to_marble import convert_model


class TinyRNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.rnn = nn.RNN(input_size=3, hidden_size=2, num_layers=1, bias=False)

    def forward(self, x, h=None):  # pragma: no cover - placeholder forward
        out, h = self.rnn(x, h)
        return out


class TwoLayerRNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.rnn = nn.RNN(input_size=3, hidden_size=2, num_layers=2, bias=False)

    def forward(self, x, h=None):  # pragma: no cover - placeholder forward
        out, h = self.rnn(x, h)
        return out


def _hidden_values(core):
    return [float(n.hidden_state) for n in core.neurons if hasattr(n, "hidden_state")]


def test_hidden_state_serialization():
    model = TinyRNN()
    core = convert_model(model)
    assert core.params.get("hidden_state_version") == 1
    assert "hidden_states" in core.params
    hs = core.params["hidden_states"]
    assert len(hs) == 1
    entry = hs[0]
    expected_device = str(model.rnn.weight_ih_l0.device)
    assert entry["layer_index"] == 0
    assert entry["direction"] == "forward"
    assert entry["shape"] == [2]
    assert entry["dtype"] == "float32"
    assert entry["device"] == expected_device
    assert entry["tensor"] == [0.0, 0.0]


devices = ["cpu"]
if torch.cuda.is_available():
    devices.append("cuda")


@pytest.mark.parametrize("device", devices)
def test_hidden_state_persistence_roundtrip(device: str):
    model = TinyRNN().to(device)
    core = convert_model(model, restore_hidden=True)
    core.params["hidden_states"][0]["tensor"] = [1.0, -1.0]
    core.params["hidden_states"][0]["device"] = device
    restore_hidden_states(core)
    json_str = core_to_json(core)
    core2 = core_from_json(json_str)
    assert _hidden_values(core2) == _hidden_values(core)
    json_str2 = core_to_json(core2)
    core3 = core_from_json(json_str2)
    assert _hidden_values(core3) == _hidden_values(core)


@pytest.mark.parametrize("device", devices)
def test_multi_layer_hidden_state_persistence(device: str):
    model = TwoLayerRNN().to(device)
    core = convert_model(model, restore_hidden=True)
    assert len(core.params["hidden_states"]) == 2
    core.params["hidden_states"][0]["tensor"] = [1.0, -1.0]
    core.params["hidden_states"][1]["tensor"] = [2.0, -2.0]
    for entry in core.params["hidden_states"]:
        entry["device"] = device
    restore_hidden_states(core)
    json_str = core_to_json(core)
    core2 = core_from_json(json_str)
    assert _hidden_values(core2) == _hidden_values(core)


@pytest.mark.parametrize("device", devices)
def test_hidden_state_corruption_raises(device: str):
    model = TinyRNN().to(device)
    core = convert_model(model)
    json_str = core_to_json(core)
    data = json.loads(json_str)
    data["hidden_states"][0]["tensor"] = [0.0]
    corrupted = json.dumps(data)
    with pytest.raises(ValueError):
        core_from_json(corrupted)


def test_hidden_state_version_roundtrip():
    model = TinyRNN()
    core = convert_model(model)
    json_str = core_to_json(core)
    data = json.loads(json_str)
    assert data["hidden_state_version"] == 1
    core2 = core_from_json(json_str)
    assert core2.params.get("hidden_state_version") == 1


def test_unknown_hidden_state_version(caplog: pytest.LogCaptureFixture):
    model = TinyRNN()
    core = convert_model(model)
    json_str = core_to_json(core)
    data = json.loads(json_str)
    data["hidden_state_version"] = 999
    corrupted = json.dumps(data)
    with caplog.at_level(logging.WARNING):
        core2 = core_from_json(corrupted)
        assert "Unknown hidden state format version" in caplog.text
    assert all(not hasattr(n, "hidden_state") for n in core2.neurons)
