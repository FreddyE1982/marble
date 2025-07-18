import json
from marble_core import Core, Neuron, Synapse


def core_to_json(core: Core) -> str:
    """Serialize a Core instance to a JSON string."""
    data = {
        "neurons": [
            {
                "id": n.id,
                "value": n.value,
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
        neuron = Neuron(n["id"], value=n["value"], tier=n.get("tier", "vram"))
        if n.get("formula"):
            neuron.formula = n["formula"]
        core.neurons.append(neuron)
    for s in payload.get("synapses", []):
        syn = Synapse(s["source"], s["target"], weight=s["weight"])
        syn.potential = s.get("potential", 1.0)
        core.synapses.append(syn)
        core.neurons[syn.source].synapses.append(syn)
    return core
