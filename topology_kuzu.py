from __future__ import annotations

"""Live synchronization of MARBLE's topology to a Kùzu graph database."""

from typing import Dict, List

from event_bus import global_event_bus
from kuzu_interface import KuzuGraphDatabase


class TopologyKuzuTracker:
    """Persist the core topology inside a Kùzu database.

    The tracker subscribes to topology change events published on the global
    event bus and mirrors those changes into a persistent Kùzu graph. The
    database is kept in sync incrementally during training so that external
    tools can inspect the evolving network structure in real time.
    """

    def __init__(self, core, db_path: str) -> None:
        self.core = core
        self.db = KuzuGraphDatabase(db_path)
        self._init_schema()
        self.rebuild()
        global_event_bus.subscribe(
            self._on_event,
            events=["neurons_added", "synapses_added", "rep_size_changed"],
        )

    # ------------------------------------------------------------------
    # schema and rebuild helpers
    # ------------------------------------------------------------------
    def _init_schema(self) -> None:
        """Ensure required node and relationship tables exist."""
        try:
            self.db.create_node_table(
                "Neuron",
                {
                    "id": "INT64",
                    "tier": "STRING",
                    "activation": "STRING",
                    "activation_flag": "BOOL",
                    "rep_size": "INT64",
                },
                "id",
            )
        except Exception:
            # Table already exists
            pass
        try:
            self.db.create_relationship_table(
                "SYNAPSE", "Neuron", "Neuron", {"weight": "DOUBLE"}
            )
        except Exception:
            pass

    def rebuild(self) -> None:
        """Rebuild the entire graph from ``self.core``."""
        # Remove existing graph
        self.db.execute("MATCH (n:Neuron) DETACH DELETE n;")
        # Add neurons
        for n in self.core.neurons:
            self.db.add_node(
                "Neuron",
                {
                    "id": n.id,
                    "tier": n.tier,
                    "activation": n.params.get("activation", ""),
                    "activation_flag": bool(n.params.get("activation_flag", False)),
                    "rep_size": self.core.rep_size,
                },
            )
        # Add synapses
        for s in self.core.synapses:
            self.db.add_relationship(
                "Neuron",
                "id",
                s.source,
                "SYNAPSE",
                "Neuron",
                "id",
                s.target,
                {"weight": s.weight},
            )

    # ------------------------------------------------------------------
    # event handling
    # ------------------------------------------------------------------
    def _on_event(self, name: str, payload: Dict) -> None:
        if name == "neurons_added":
            for n in payload.get("neurons", []):
                self.db.add_node("Neuron", n)
        elif name == "synapses_added":
            for s in payload.get("synapses", []):
                self.db.add_relationship(
                    "Neuron",
                    "id",
                    s["src"],
                    "SYNAPSE",
                    "Neuron",
                    "id",
                    s["dst"],
                    {"weight": s["weight"]},
                )
        elif name == "rep_size_changed":
            new_size = payload.get("new_size")
            if new_size is not None:
                self.db.execute(
                    "MATCH (n:Neuron) SET n.rep_size = $size;", {"size": new_size}
                )


__all__ = ["TopologyKuzuTracker"]
