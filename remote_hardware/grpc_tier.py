"""gRPC-based remote tier implementation."""

from __future__ import annotations

import time
import grpc

from .base import RemoteTier


class GrpcRemoteTier(RemoteTier):
    """Remote tier using a simple gRPC unary API with retry support."""

    def __init__(
        self,
        address: str,
        *,
        max_retries: int = 3,
        backoff_factor: float = 0.5,
    ) -> None:
        super().__init__(address)
        self.channel: grpc.Channel | None = None
        self.stub = None
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor

    def connect(self) -> None:
        """Create gRPC channel and prepare stub."""
        if self.channel:
            return
        self.channel = grpc.insecure_channel(self.address)
        self.stub = self.channel.unary_unary("/Remote/Process")

    def offload_core(self, core_bytes: bytes) -> bytes:
        """Send ``core_bytes`` to the remote tier and return response.

        Retries transient ``grpc.RpcError`` failures using an exponential
        backoff schedule. Each failed attempt closes the channel to release GPU
        resources before trying again.
        """
        if self.stub is None:
            self.connect()
        assert self.stub is not None

        for attempt in range(self.max_retries):
            try:
                return self.stub(core_bytes)
            except grpc.RpcError:
                self.close()
                if attempt == self.max_retries - 1:
                    raise
                time.sleep(self.backoff_factor * (2**attempt))
                self.connect()

    def close(self) -> None:
        """Close the gRPC channel."""
        if self.channel:
            self.channel.close()
            self.channel = None
            self.stub = None


def get_remote_tier(
    address: str = "localhost:50051",
    *,
    max_retries: int = 3,
    backoff_factor: float = 0.5,
) -> GrpcRemoteTier:
    """Factory used by plugin loader."""
    return GrpcRemoteTier(
        address, max_retries=max_retries, backoff_factor=backoff_factor
    )
