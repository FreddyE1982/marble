"""Utilities for inspecting neural pathways with CPU/GPU support."""

from __future__ import annotations

import networkx as nx
import torch
from plotly import graph_objs as go

from networkx_interop import core_to_networkx


def find_neural_pathway(core, start_id: int, end_id: int, device: str | None = None) -> list[int]:
    """Return the neuron id path from ``start_id`` to ``end_id``.

    The computation uses a tensor-based breadth first search that runs on the
    specified ``device`` (``"cuda"`` when available, otherwise ``"cpu"``).
    """
    g = core_to_networkx(core)
    nodes = list(g.nodes())
    if start_id not in nodes or end_id not in nodes:
        return []
    idx_map = {n: i for i, n in enumerate(nodes)}
    n = len(nodes)
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    adj = torch.zeros((n, n), device=dev, dtype=torch.bool)
    for u, v in g.edges():
        adj[idx_map[u], idx_map[v]] = True
    start_idx, end_idx = idx_map[start_id], idx_map[end_id]
    if start_idx == end_idx:
        return [start_id]
    visited = torch.zeros(n, dtype=torch.bool, device=dev)
    prev = torch.full((n,), -1, dtype=torch.long, device=dev)
    queue = [start_idx]
    visited[start_idx] = True
    while queue:
        cur = queue.pop(0)
        if cur == end_idx:
            break
        neighbors = torch.where(adj[cur])[0].tolist()
        for nb in neighbors:
            if not visited[nb]:
                visited[nb] = True
                prev[nb] = cur
                queue.append(nb)
    if not visited[end_idx]:
        return []
    path_idx = []
    cur = end_idx
    while cur != -1:
        path_idx.append(cur)
        cur = prev[cur].item()
    path_idx.reverse()
    return [nodes[i] for i in path_idx]


def pathway_figure(core, path: list[int], layout: str = "spring") -> go.Figure:
    """Return a Plotly figure highlighting ``path`` on the core graph."""
    g = core_to_networkx(core)
    if layout == "circular":
        pos = nx.circular_layout(g)
    else:
        pos = nx.spring_layout(g, seed=42)
    path_edges = set(zip(path, path[1:]))
    edge_x, edge_y, edge_color = [], [], []
    for u, v in g.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]
        color = "red" if (u, v) in path_edges else "#888"
        edge_color.append(color)
    node_x, node_y, node_color = [], [], []
    for n in g.nodes():
        x, y = pos[n]
        node_x.append(x)
        node_y.append(y)
        node_color.append("red" if n in path else "blue")
    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        mode="lines",
        line=dict(width=1, color="#888"),
        hoverinfo="none",
    )
    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers",
        hoverinfo="text",
        marker=dict(size=8, color=node_color),
    )
    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(showlegend=False, margin=dict(l=0, r=0, t=0, b=0))
    return fig
