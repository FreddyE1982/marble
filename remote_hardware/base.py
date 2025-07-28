"""Base classes for remote hardware plugins."""

from __future__ import annotations

from abc import ABC, abstractmethod


class RemoteTier(ABC):
    """Abstract base class for remote compute tiers."""

    name: str = "remote"

    def __init__(self, address: str) -> None:
        self.address = address

    @abstractmethod
    def connect(self) -> None:
        """Establish connection to remote hardware."""

    @abstractmethod
    def offload_core(self, core_bytes: bytes) -> bytes:
        """Send serialized core state and return processed state."""

    @abstractmethod
    def close(self) -> None:
        """Close any open network connections."""
