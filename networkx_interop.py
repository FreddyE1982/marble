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


def core_to_dict(core: Core) -> dict:
    """Serialise ``core`` into a plain Python ``dict``.

    The returned dictionary follows the structure::

        {
            "nodes": [
                {"id": int, "value": float, "tier": str, "neuron_type": str},
                ...
            ],
            "edges": [
                {"source": int, "target": int, "weight": float, "synapse_type": str},
                ...
            ],
        }

    The export is device agnostic – tensor data remains on its
    original device (CPU or GPU) and only lightweight metadata is
    captured.  This makes the representation suitable for transfer to
    visualisation frontends.
    """

    nodes = [
        {
            "id": n.id,
            "value": n.value,
            "tier": n.tier,
            "neuron_type": n.neuron_type,
        }
        for n in core.neurons
    ]

    edges = [
        {
            "source": s.source,
            "target": s.target,
            "weight": s.weight,
            "synapse_type": s.synapse_type,
        }
        for s in core.synapses
    ]

    return {"nodes": nodes, "edges": edges}


def core_diff(prev: dict, core: Core) -> dict:
    """Return incremental updates between ``prev`` snapshot and ``core``.

    Parameters
    ----------
    prev:
        Previous snapshot produced by :func:`core_to_dict`.
    core:
        Current :class:`Core` instance to diff against ``prev``.

    Returns
    -------
    dict
        Dictionary containing lists ``added_nodes``, ``removed_nodes``,
        ``updated_nodes``, ``added_edges``, ``removed_edges`` and
        ``updated_edges``. Each list mirrors the structure used by
        :func:`core_to_dict`.  The operation inspects only lightweight
        metadata and leaves tensor data on its original device, making it
        safe for both CPU and GPU execution.
    """

    current = core_to_dict(core)

    prev_nodes = {n["id"]: n for n in prev.get("nodes", [])}
    curr_nodes = {n["id"]: n for n in current.get("nodes", [])}

    added_nodes = [curr_nodes[i] for i in curr_nodes.keys() - prev_nodes.keys()]
    removed_nodes = [prev_nodes[i] for i in prev_nodes.keys() - curr_nodes.keys()]
    updated_nodes = [
        curr_nodes[i]
        for i in curr_nodes.keys() & prev_nodes.keys()
        if curr_nodes[i] != prev_nodes[i]
    ]

    prev_edges = {
        (e["source"], e["target"]): e for e in prev.get("edges", [])
    }
    curr_edges = {
        (e["source"], e["target"]): e for e in current.get("edges", [])
    }

    added_edges = [curr_edges[k] for k in curr_edges.keys() - prev_edges.keys()]
    removed_edges = [prev_edges[k] for k in prev_edges.keys() - curr_edges.keys()]
    updated_edges = [
        curr_edges[k]
        for k in curr_edges.keys() & prev_edges.keys()
        if curr_edges[k] != prev_edges[k]
    ]

    return {
        "added_nodes": added_nodes,
        "removed_nodes": removed_nodes,
        "updated_nodes": updated_nodes,
        "added_edges": added_edges,
        "removed_edges": removed_edges,
        "updated_edges": updated_edges,
    }


def dict_to_core(data: dict, params: dict) -> Core:
    """Reconstruct a :class:`Core` from ``data`` produced by
    :func:`core_to_dict`.

    Parameters
    ----------
    data:
        Dictionary with ``nodes`` and ``edges`` lists.
    params:
        Parameters forwarded to :class:`Core` during initialisation.
    """

    core = Core(params)
    core.neurons = []
    core.synapses = []
    node_map: dict[int, Neuron] = {}
    for n in data.get("nodes", []):
        neuron = Neuron(
            n["id"],
            value=n.get("value", 0.0),
            tier=n.get("tier", "vram"),
            neuron_type=n.get("neuron_type", "standard"),
            rep_size=params.get("representation_size", core.rep_size),
        )
        core.neurons.append(neuron)
        node_map[neuron.id] = neuron

    for e in data.get("edges", []):
        syn = Synapse(
            e["source"],
            e["target"],
            weight=e.get("weight", 1.0),
            synapse_type=e.get("synapse_type", "standard"),
        )
        core.synapses.append(syn)
        node_map[e["source"]].synapses.append(syn)

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
