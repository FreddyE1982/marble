"""Tests for remote wanderer message helpers."""
import torch

from message_bus import MessageBus
from wanderer_messages import PathUpdate
from remote_wanderer import (
    current_device,
    send_exploration_request,
    receive_exploration_request,
    send_exploration_result,
    receive_exploration_result,
)


def test_request_device_transmission():
    bus = MessageBus()
    bus.register("coordinator")
    bus.register("wanderer")

    send_exploration_request(
        bus,
        "coordinator",
        "wanderer",
        seed=42,
        max_steps=5,
    )
    msg = bus.receive("wanderer")
    req = receive_exploration_request(msg)
    assert req.device == current_device()


def test_result_device_transmission():
    bus = MessageBus()
    bus.register("coordinator")
    bus.register("wanderer")

    paths = [PathUpdate(nodes=[1, 2], score=0.5)]
    send_exploration_result(bus, "wanderer", "coordinator", paths)
    msg = bus.receive("coordinator")
    res = receive_exploration_result(msg)
    assert res.device == current_device()
    assert len(res.paths) == 1 and res.paths[0].nodes == [1, 2]
