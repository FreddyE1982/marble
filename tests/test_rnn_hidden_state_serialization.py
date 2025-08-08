import torch
from torch import nn

from pytorch_to_marble import convert_model


class TinyRNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.RNN(input_size=3, hidden_size=2, num_layers=1, bias=False)

    def forward(self, x, h=None):  # pragma: no cover - placeholder forward
        out, h = self.rnn(x, h)
        return out


def test_hidden_state_serialization():
    model = TinyRNN()
    core = convert_model(model)
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
