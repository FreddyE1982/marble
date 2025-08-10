import os
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from pytorch_to_marble import convert_model, register_converter, LAYER_CONVERTERS


class DoubleLinear(torch.nn.Module):
    """Linear layer followed by a fixed scaling factor of 2."""

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.fc = torch.nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - example script
        return self.fc(x) * 2.0


@register_converter(DoubleLinear)
def convert_double_linear(layer: DoubleLinear, core, inputs):
    """Convert :class:`DoubleLinear` into MARBLE neurons."""
    out_ids = LAYER_CONVERTERS[torch.nn.Linear](layer.fc, core, inputs)
    for nid in out_ids:
        for syn in core.neurons[nid].synapses:
            syn.weight *= 2.0
    return out_ids


class WrapperModel(torch.nn.Module):
    """Model that uses the custom :class:`DoubleLinear` layer."""

    def __init__(self) -> None:
        super().__init__()
        self.double_layer = DoubleLinear(4, 3)
        self.input_size = 4

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - example script
        return self.double_layer(x)


def main() -> None:  # pragma: no cover - example script
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = WrapperModel().to(device)
    sample = torch.randn(1, 4, device=device)
    traced_out = model(sample)
    core = convert_model(model)
    print("PyTorch output:", traced_out.cpu().tolist())
    print("Converted neurons:", len(core.neurons))
    print("Converted synapses:", len(core.synapses))


if __name__ == "__main__":  # pragma: no cover - example script
    main()
