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


def pipeline_to_networkx(pipeline: list[dict]) -> nx.DiGraph:
    """Convert a pipeline description into a ``networkx`` graph.

    Each step in ``pipeline`` becomes a node. Edges are created from
    dependencies to dependents. Macros and branches are expanded
    recursively with node names prefixed by their parents to keep them
    unique. Sequential steps without explicit dependencies are connected
    linearly.
    """

    graph = nx.DiGraph()

    def add_steps(steps: list[dict], prefix: str = "") -> tuple[str, str]:
        prev_name: str | None = None
        first_name: str | None = None
        for idx, step in enumerate(steps):
            base = step.get("name") or f"step_{idx}"
            name = f"{prefix}{base}"
            label = step.get("plugin") or step.get("func")
            if label is None:
                label = "macro" if "macro" in step else "branch" if "branches" in step else "step"
            graph.add_node(name, label=label, params=step.get("params", {}))
            deps = [f"{prefix}{d}" for d in step.get("depends_on", [])]
            if not deps and prev_name is not None:
                deps = [prev_name]
            for dep in deps:
                graph.add_edge(dep, name)
            if first_name is None:
                first_name = name
            if "macro" in step:
                sub_first, sub_last = add_steps(step["macro"], prefix=f"{name}::")
                graph.add_edge(name, sub_first)
                prev_name = sub_last
            elif "branches" in step:
                branch_last = []
                for b_idx, branch in enumerate(step["branches"]):
                    b_first, b_last = add_steps(branch, prefix=f"{name}::b{b_idx}::")
                    graph.add_edge(name, b_first)
                    branch_last.append(b_last)
                merge_spec = step.get("merge")
                if merge_spec:
                    merge_name = merge_spec.get("name") or "merge"
                    merge_full = f"{name}::{merge_name}"
                    merge_label = merge_spec.get("plugin") or merge_spec.get("func") or "merge"
                    graph.add_node(merge_full, label=merge_label, params=merge_spec.get("params", {}))
                    for last in branch_last:
                        graph.add_edge(last, merge_full)
                    prev_name = merge_full
                else:
                    prev_name = name
            else:
                prev_name = name
        assert first_name is not None
        return first_name, prev_name  # type: ignore

    add_steps(pipeline)
    return graph


def pipeline_to_core(pipeline: list[dict], params: dict) -> Core:
    """Build a :class:`Core` representation of ``pipeline``."""

    graph = pipeline_to_networkx(pipeline)
    return networkx_to_core(graph, params)
