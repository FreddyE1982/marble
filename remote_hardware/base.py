"""Base classes for remote hardware plugins."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch


class RemoteTier(ABC):
    """Abstract base class for remote compute tiers.

    Subclasses provide the mechanism for connecting to an external piece of
    hardware and executing work there.  In addition to offloading entire core
    state blobs, tiers may be asked to run individual pipeline steps via
    :meth:`run_step`.  Implementations should honour the provided ``device``
    argument so steps execute on CPU or GPU consistently with local execution.
    """

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
    def run_step(
        self, step: dict, marble: Any | None, device: torch.device
    ) -> Any:
        """Execute ``step`` on the remote tier and return its result."""

    @abstractmethod
    def close(self) -> None:
        """Close any open network connections."""
