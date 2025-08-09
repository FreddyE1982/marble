"""Tests for wanderer message serialisation."""
from wanderer_messages import ExplorationRequest, ExplorationResult, PathUpdate


def test_request_round_trip() -> None:
    req = ExplorationRequest(wanderer_id="w1", seed=123, max_steps=10, device="cuda:0", timestamp=1.0)
    payload = req.to_payload()
    restored = ExplorationRequest.from_payload(payload)
    assert restored == req


def test_result_round_trip() -> None:
    paths = [PathUpdate(nodes=[1, 2, 3], score=0.5), PathUpdate(nodes=[4, 5], score=1.2)]
    res = ExplorationResult(wanderer_id="w2", paths=paths, device="cpu", timestamp=2.0)
    payload = res.to_payload()
    restored = ExplorationResult.from_payload(payload)
    assert restored == res
