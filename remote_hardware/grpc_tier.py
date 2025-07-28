"""gRPC-based remote tier implementation."""

from __future__ import annotations

import grpc

from .base import RemoteTier


class GrpcRemoteTier(RemoteTier):
    """Remote tier using a simple gRPC unary API."""

    def __init__(self, address: str) -> None:
        super().__init__(address)
        self.channel: grpc.Channel | None = None
        self.stub = None

    def connect(self) -> None:
        """Create gRPC channel and prepare stub."""
        if self.channel:
            return
        self.channel = grpc.insecure_channel(self.address)
        self.stub = self.channel.unary_unary("/Remote/Process")

    def offload_core(self, core_bytes: bytes) -> bytes:
        """Send ``core_bytes`` to the remote tier and return response."""
        if self.stub is None:
            self.connect()
        assert self.stub is not None
        return self.stub(core_bytes)

    def close(self) -> None:
        """Close the gRPC channel."""
        if self.channel:
            self.channel.close()
            self.channel = None
            self.stub = None


def get_remote_tier(address: str = "localhost:50051") -> GrpcRemoteTier:
    """Factory used by plugin loader."""
    return GrpcRemoteTier(address)
