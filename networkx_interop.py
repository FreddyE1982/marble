import networkx as nx

from marble_core import Core, Neuron, Synapse


def core_to_networkx(core: Core) -> nx.DiGraph:
    """Convert a :class:`Core` to a directed NetworkX graph."""
    graph = nx.DiGraph()
    for neuron in core.neurons:
        graph.add_node(
            neuron.id,
            **{
                "value": neuron.value,
                "tier": neuron.tier,
                "neuron_type": neuron.neuron_type,
            }
        )
    for syn in core.synapses:
        graph.add_edge(
            syn.source, syn.target, weight=syn.weight, synapse_type=syn.synapse_type
        )
    return graph


def networkx_to_core(graph: nx.DiGraph, params: dict) -> Core:
    """Create a :class:`Core` from a NetworkX graph."""
    core = Core(params)
    core.neurons = []
    core.synapses = []
    node_map: dict[int, Neuron] = {}
    for node_id, data in graph.nodes(data=True):
        n = Neuron(
            node_id,
            value=data.get("value", 0.0),
            tier=data.get("tier", "vram"),
            neuron_type=data.get("neuron_type", "standard"),
            rep_size=params.get("representation_size", core.rep_size),
        )
        core.neurons.append(n)
        node_map[node_id] = n
    for src, tgt, data in graph.edges(data=True):
        syn = Synapse(
            src,
            tgt,
            weight=data.get("weight", 1.0),
            synapse_type=data.get("synapse_type", "standard"),
        )
        core.synapses.append(syn)
        node_map[src].synapses.append(syn)
    return core
