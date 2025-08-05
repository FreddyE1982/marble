from __future__ import annotations

"""Utility functions for visualising neuron graphs.

This module provides a reusable component that converts the JSON-style graph
representation returned by :func:`networkx_interop.core_to_dict` or the
``/graph`` API endpoint into an interactive Plotly Sankey diagram.  The helper
supports filtering edges by weight and nodes by degree to aid exploration of
large graphs.  All operations are performed using plain Python data structures
and therefore run identically on CPU and GPU systems.
"""

from typing import Dict, List

import plotly.graph_objects as go


def sankey_figure(
    data: Dict,
    weight_threshold: float = 0.0,
    degree_threshold: int = 0,
) -> go.Figure:
    """Return a Plotly Sankey figure visualising ``data``.

    Parameters
    ----------
    data:
        Dictionary containing ``nodes`` and ``edges`` lists as produced by
        :func:`networkx_interop.core_to_dict` or the ``/graph`` API endpoint.
    weight_threshold:
        Minimum absolute edge weight required for inclusion.
    degree_threshold:
        Minimum combined in/out degree a node must have to be retained.

    Returns
    -------
    go.Figure
        Sankey diagram showing the filtered neuron graph.
    """

    nodes: List[Dict] = data.get("nodes", [])
    edges: List[Dict] = data.get("edges", [])

    # Filter edges by weight
    filtered_edges = [
        e for e in edges if abs(float(e.get("weight", 0.0))) >= weight_threshold
    ]

    # Compute degree counts from remaining edges
    degree: Dict[int, int] = {}
    for e in filtered_edges:
        degree[e["source"]] = degree.get(e["source"], 0) + 1
        degree[e["target"]] = degree.get(e["target"], 0) + 1

    # Filter nodes by degree and build index mapping
    filtered_nodes = [n for n in nodes if degree.get(n["id"], 0) >= degree_threshold]
    id_to_index = {n["id"]: i for i, n in enumerate(filtered_nodes)}

    # Remove edges that reference dropped nodes
    filtered_edges = [
        e
        for e in filtered_edges
        if e["source"] in id_to_index and e["target"] in id_to_index
    ]

    labels = [str(n["id"]) for n in filtered_nodes]
    sources = [id_to_index[e["source"]] for e in filtered_edges]
    targets = [id_to_index[e["target"]] for e in filtered_edges]
    values = [abs(float(e.get("weight", 0.0))) for e in filtered_edges]

    node = dict(label=labels, pad=10, thickness=10)
    link = dict(source=sources, target=targets, value=values)

    return go.Figure(data=[go.Sankey(node=node, link=link)])
