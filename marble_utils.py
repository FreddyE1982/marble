import json
import numpy as np
from marble_core import Core, Neuron, Synapse, _W1, _B1, _W2, _B2
import torch

def get_default_device() -> torch.device:
    """Return CUDA device if available else CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")



def core_to_json(core: Core) -> str:
    """Serialize a Core instance to a JSON string."""
    data = {
        "neurons": [
            {
                "id": n.id,
                "value": n.value.tolist() if hasattr(n.value, "tolist") else n.value,
                "tier": n.tier,
                "formula": str(n.formula) if n.formula is not None else None,
            }
            for n in core.neurons
        ],
        "synapses": [
            {
                "source": s.source,
                "target": s.target,
                "weight": s.weight,
                "potential": s.potential,
                "synapse_type": s.synapse_type,
            }
            for s in core.synapses
        ],
    }
    return json.dumps(data)


def core_from_json(json_str: str) -> Core:
    """Create a Core instance from a JSON string."""
    payload = json.loads(json_str)
    params = {
        'xmin': -2.0,
        'xmax': 1.0,
        'ymin': -1.5,
        'ymax': 1.5,
        'width': 1,
        'height': 1,
        'max_iter': 1,
        'vram_limit_mb': 0.1,
        'ram_limit_mb': 0.1,
        'disk_limit_mb': 0.1,
    }
    core = Core(params, formula=None, formula_num_neurons=0)
    core.neurons = []
    core.synapses = []
    for n in payload.get("neurons", []):
        val = n["value"]
        if isinstance(val, list):
            val = np.array(val)
        neuron = Neuron(n["id"], value=val, tier=n.get("tier", "vram"))
        if n.get("formula"):
            neuron.formula = n["formula"]
        core.neurons.append(neuron)
    for s in payload.get("synapses", []):
        syn = Synapse(
            s["source"],
            s["target"],
            weight=s["weight"],
            synapse_type=s.get("synapse_type", "standard"),
        )
        syn.potential = s.get("potential", 1.0)
        core.synapses.append(syn)
        core.neurons[syn.source].synapses.append(syn)
    return core


def export_neuron_state(core: Core, path: str) -> None:
    """Persist neuron states to disk for later restoration."""
    data = [n.representation.tolist() for n in core.neurons]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)


def import_neuron_state(core: Core, path: str) -> None:
    """Load neuron states from ``path`` into ``core``."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if len(data) != len(core.neurons):
        raise ValueError("Neuron count mismatch when importing state")
    for rep, neuron in zip(data, core.neurons):
        neuron.representation = np.asarray(rep, dtype=float)


def export_core_to_onnx(core: Core, path: str) -> None:
    """Export the message passing MLP to an ONNX file."""
    import torch

    class MPModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.w1 = torch.nn.Parameter(torch.tensor(_W1, dtype=torch.float32))
            self.b1 = torch.nn.Parameter(torch.tensor(_B1, dtype=torch.float32))
            self.w2 = torch.nn.Parameter(torch.tensor(_W2, dtype=torch.float32))
            self.b2 = torch.nn.Parameter(torch.tensor(_B2, dtype=torch.float32))

        def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover
            h = torch.tanh(x @ self.w1 + self.b1)
            return torch.tanh(h @ self.w2 + self.b2)

    device = get_default_device()
    model = MPModel().to(device)
    dummy = torch.randn(len(core.neurons), core.rep_size, device=device)
    torch.onnx.export(model, dummy, path, input_names=["x"], output_names=["out"], opset_version=17)


