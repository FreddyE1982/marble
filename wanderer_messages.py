"""Message schemas for remote wanderer coordination.

Provides dataclasses that can be serialized to dictionaries for
transmission over the :class:`MessageBus`. These messages carry
exploration commands and results between the coordinator and remote
wanderers. All messages include the execution ``device`` so that
receivers can route tensors to CPU or GPU appropriately.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List
import time


@dataclass
class ExplorationRequest:
    """Command issued by the coordinator to start exploration.

    Attributes
    ----------
    wanderer_id:
        Identifier of the remote wanderer that should execute the request.
    seed:
        Random seed to ensure deterministic behaviour across devices.
    max_steps:
        Maximum number of steps the wanderer may take.
    device:
        Execution device, e.g. ``"cpu"`` or ``"cuda:0"``.
    timestamp:
        Time the request was created (``time.time()``).
    """

    wanderer_id: str
    seed: int
    max_steps: int
    device: str
    timestamp: float = time.time()

    def to_payload(self) -> Dict[str, Any]:
        """Return a serialisable dictionary representing the request."""
        return {
            "wanderer_id": self.wanderer_id,
            "seed": self.seed,
            "max_steps": self.max_steps,
            "device": self.device,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_payload(cls, payload: Dict[str, Any]) -> "ExplorationRequest":
        """Create an instance from a payload dictionary."""
        return cls(
            wanderer_id=str(payload["wanderer_id"]),
            seed=int(payload["seed"]),
            max_steps=int(payload["max_steps"]),
            device=str(payload.get("device", "cpu")),
            timestamp=float(payload.get("timestamp", time.time())),
        )


@dataclass
class PathUpdate:
    """Single exploration path discovered by a wanderer.

    Attributes
    ----------
    nodes:
        Sequence of visited node identifiers.
    score:
        Numeric score associated with the path.
    """

    nodes: List[int]
    score: float

    def to_payload(self) -> Dict[str, Any]:
        return {"nodes": self.nodes, "score": self.score}

    @classmethod
    def from_payload(cls, payload: Dict[str, Any]) -> "PathUpdate":
        return cls(nodes=list(payload["nodes"]), score=float(payload["score"]))


@dataclass
class ExplorationResult:
    """Result message sent from a wanderer to the coordinator.

    Attributes
    ----------
    wanderer_id:
        Identifier of the sending wanderer.
    paths:
        List of :class:`PathUpdate` instances representing exploration
        results.
    device:
        Execution device used by the wanderer.
    timestamp:
        Time the result was created (``time.time()``).
    """

    wanderer_id: str
    paths: List[PathUpdate]
    device: str
    timestamp: float = time.time()

    def to_payload(self) -> Dict[str, Any]:
        """Return a serialisable dictionary representing the result."""
        return {
            "wanderer_id": self.wanderer_id,
            "paths": [p.to_payload() for p in self.paths],
            "device": self.device,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_payload(cls, payload: Dict[str, Any]) -> "ExplorationResult":
        return cls(
            wanderer_id=str(payload["wanderer_id"]),
            paths=[PathUpdate.from_payload(p) for p in payload.get("paths", [])],
            device=str(payload.get("device", "cpu")),
            timestamp=float(payload.get("timestamp", time.time())),
        )


__all__ = [
    "ExplorationRequest",
    "ExplorationResult",
    "PathUpdate",
]
