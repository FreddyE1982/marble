import grpc
import pytest

from remote_hardware.grpc_tier import GrpcRemoteTier


class DummyTier(GrpcRemoteTier):
    """GrpcRemoteTier with injectable failure pattern for testing."""

    def __init__(self, fails: int):
        super().__init__("addr", max_retries=3, backoff_factor=0.0)
        self._fails = fails
        self.closed = 0

    def connect(self) -> None:  # type: ignore[override]
        def stub(data: bytes) -> bytes:
            if self._fails > 0:
                self._fails -= 1
                raise grpc.RpcError("temp")
            return b"ok"

        self.stub = stub

    def close(self) -> None:  # type: ignore[override]
        self.closed += 1
        self.stub = None


def test_grpc_tier_retry_success():
    tier = DummyTier(fails=1)
    result = tier.offload_core(b"core")
    assert result == b"ok"
    assert tier.closed >= 1  # channel closed on failure


def test_grpc_tier_persistent_failure():
    tier = DummyTier(fails=5)
    with pytest.raises(grpc.RpcError):
        tier.offload_core(b"core")
    assert tier.closed >= 3
