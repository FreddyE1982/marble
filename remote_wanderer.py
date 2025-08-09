"""Utilities for exchanging exploration messages with device metadata.

These helpers wrap :class:`MessageBus` to ensure that every payload
includes the sender's execution device (CPU or GPU).  This allows
coordinator and wanderers to route tensors appropriately when working in
heterogeneous environments.
"""
from __future__ import annotations

from typing import Iterable

import torch

from message_bus import MessageBus, Message
from wanderer_messages import ExplorationRequest, ExplorationResult, PathUpdate

__all__ = [
    "current_device",
    "send_exploration_request",
    "receive_exploration_request",
    "send_exploration_result",
    "receive_exploration_result",
]


def current_device() -> str:
    """Return the current execution device.

    Detects GPU availability and returns a device string that can be used in
    payloads.  The string uses the standard ``cuda:0`` format for GPUs.
    """

    return "cuda:0" if torch.cuda.is_available() else "cpu"


def send_exploration_request(
    bus: MessageBus,
    sender: str,
    recipient: str,
    *,
    seed: int,
    max_steps: int,
    device: str | None = None,
) -> None:
    """Send an :class:`ExplorationRequest` to ``recipient``.

    Parameters
    ----------
    bus:
        Message bus instance used for transmission.
    sender:
        Identifier of the sending agent (usually the coordinator).
    recipient:
        Identifier of the remote wanderer.
    seed:
        Random seed controlling exploration behaviour.
    max_steps:
        Maximum number of steps the wanderer may take.
    device:
        Optional device string.  If ``None`` the current device is detected
        automatically.
    """

    if device is None:
        device = current_device()
    request = ExplorationRequest(
        wanderer_id=recipient,
        seed=seed,
        max_steps=max_steps,
        device=device,
    )
    bus.send(sender, recipient, request.to_payload())


def receive_exploration_request(msg: Message) -> ExplorationRequest:
    """Decode an :class:`ExplorationRequest` from ``msg``."""

    return ExplorationRequest.from_payload(msg.content)


def send_exploration_result(
    bus: MessageBus,
    sender: str,
    recipient: str,
    paths: Iterable[PathUpdate],
    *,
    device: str | None = None,
) -> None:
    """Send an :class:`ExplorationResult` to ``recipient``.

    Parameters
    ----------
    bus:
        Message bus instance used for transmission.
    sender:
        Identifier of the wanderer sending the results.
    recipient:
        Identifier of the coordinator receiving the results.
    paths:
        Iterable of :class:`PathUpdate` objects describing discovered paths.
    device:
        Optional execution device of the wanderer.  Detected automatically when
        omitted.
    """

    if device is None:
        device = current_device()
    result = ExplorationResult(wanderer_id=sender, paths=list(paths), device=device)
    bus.send(sender, recipient, result.to_payload())


def receive_exploration_result(msg: Message) -> ExplorationResult:
    """Decode an :class:`ExplorationResult` from ``msg``."""

    return ExplorationResult.from_payload(msg.content)
