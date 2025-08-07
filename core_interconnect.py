import copy
import random
from marble_core import Core, Synapse


def interconnect_cores(cores: list[Core], prob: float | None = None) -> Core:
    """Return a new :class:`Core` combining ``cores`` with interconnection synapses.

    Parameters
    ----------
    cores:
        List of cores to merge.
    prob:
        Optional probability of creating an interconnection synapse between any
        pair of neurons belonging to different cores. When ``None`` the average
        ``interconnection_prob`` from the provided cores' parameters is used.
    """
    if not cores:
        raise ValueError("No cores provided")

    if prob is None:
        probs = [c.params.get("interconnection_prob", 0.05) for c in cores]
        prob = sum(probs) / len(probs)

    # clone first core as base
    base_params = cores[0].params.copy()
    combined = Core(base_params, formula=None, formula_num_neurons=0)
    combined.neurons = []
    combined.synapses = []
    offsets = []
    offset = 0
    for c in cores:
        offsets.append(offset)
        for n in c.neurons:
            new_n = copy.deepcopy(n)
            new_n.id = offset + n.id
            new_n.synapses = []
            combined.neurons.append(new_n)
        for s in c.synapses:
            syn = Synapse(
                s.source + offset,
                s.target + offset,
                weight=s.weight,
                synapse_type=s.synapse_type,
                frozen=s.frozen,
                echo_length=len(s.echo_buffer),
            )
            combined.synapses.append(syn)
            combined.neurons[syn.source].synapses.append(syn)
        offset += len(c.neurons)

    # interconnect cores
    total = len(combined.neurons)
    if len(cores) > 1 and prob > 0.0:
        for i, c1 in enumerate(cores):
            off1 = offsets[i]
            for j, c2 in enumerate(cores):
                if j <= i:
                    continue
                off2 = offsets[j]
                for n1 in range(len(c1.neurons)):
                    for n2 in range(len(c2.neurons)):
                        if random.random() < prob:
                            syn = Synapse(
                                off1 + n1,
                                off2 + n2,
                                weight=random.uniform(0.5, 1.5),
                                synapse_type="interconnection",
                                remote_core=cores[j],
                                remote_target=n2,
                            )
                            combined.synapses.append(syn)
                            combined.neurons[syn.source].synapses.append(syn)
    return combined

