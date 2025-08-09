"""Utilities for exchanging exploration messages with device metadata.

These helpers wrap :class:`MessageBus` to ensure that every payload
includes the sender's execution device (CPU or GPU).  This allows
coordinator and wanderers to route tensors appropriately when working in
heterogeneous environments.
"""

from __future__ import annotations

import time
from typing import Callable, Iterable

import torch

from message_bus import AsyncDispatcher, Message, MessageBus
from wanderer_messages import ExplorationRequest, ExplorationResult, PathUpdate

__all__ = [
    "current_device",
    "send_exploration_request",
    "receive_exploration_request",
    "send_exploration_result",
    "receive_exploration_result",
    "RemoteWandererClient",
    "RemoteWandererServer",
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
    delay: float = 0.0,
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
    delay:
        Artificial delay in seconds before dispatching the request. Useful for
        simulating network latency during benchmarks.
    """

    if device is None:
        device = current_device()
    request = ExplorationRequest(
        wanderer_id=recipient,
        seed=seed,
        max_steps=max_steps,
        device=device,
    )
    if delay:
        time.sleep(delay)
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
    delay: float = 0.0,
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
    delay:
        Artificial delay in seconds before sending the result. Allows latency
        simulations.
    """

    if device is None:
        device = current_device()
    result = ExplorationResult(wanderer_id=sender, paths=list(paths), device=device)
    if delay:
        time.sleep(delay)
    bus.send(sender, recipient, result.to_payload())


def receive_exploration_result(msg: Message) -> ExplorationResult:
    """Decode an :class:`ExplorationResult` from ``msg``."""

    return ExplorationResult.from_payload(msg.content)


class RemoteWandererClient:
    """Client handling exploration requests for a remote wanderer.

    The client listens on its dedicated :class:`MessageBus` queue and reacts to
    incoming :class:`ExplorationRequest` messages.  For each request the provided
    ``explore`` callback is invoked to perform the actual wandering logic.  The
    resulting paths are returned to the coordinator via
    :class:`ExplorationResult` messages.  Device information is transmitted with
    every payload so coordinators can route tensors correctly on CPU or GPU.
    A configurable ``network_latency`` can be supplied to simulate slow network
    links for benchmarking purposes.
    """

    def __init__(
        self,
        bus: MessageBus,
        wanderer_id: str,
        explore: Callable[[int, int], Iterable[PathUpdate]],
        *,
        device: str | None = None,
        poll_interval: float = 0.1,
        network_latency: float = 0.0,
    ) -> None:
        self._bus = bus
        self._wanderer_id = wanderer_id
        self._explore = explore
        self._device = device
        self._latency = network_latency
        self._dispatcher = AsyncDispatcher(
            bus, wanderer_id, self._handle_request, poll_interval=poll_interval
        )

    # ------------------------------------------------------------------
    def start(self) -> None:
        """Register with the bus and begin processing requests."""

        self._bus.register(self._wanderer_id)
        self._dispatcher.start()

    # ------------------------------------------------------------------
    def stop(self) -> None:
        """Stop processing requests."""

        self._dispatcher.stop()

    # ------------------------------------------------------------------
    def _handle_request(self, msg: Message) -> None:
        req = receive_exploration_request(msg)
        torch.manual_seed(req.seed)
        try:
            paths = list(self._explore(req.seed, req.max_steps, req.device))
        except TypeError:
            paths = list(self._explore(req.seed, req.max_steps))
        recipient = msg.sender or ""
        send_exploration_result(
            self._bus,
            self._wanderer_id,
            recipient,
            paths,
            device=self._device or current_device(),
            delay=self._latency,
        )


class RemoteWandererServer:
    """Coordinator dispatching exploration requests to remote wanderers.

    The server optionally delays outgoing requests using ``network_latency`` to
    emulate slow communication channels during benchmarks.
    """

    def __init__(
        self,
        bus: MessageBus,
        coordinator_id: str,
        *,
        timeout: float = 5.0,
        network_latency: float = 0.0,
    ) -> None:
        self._bus = bus
        self._coordinator_id = coordinator_id
        self._timeout = timeout
        self._latency = network_latency
        self._bus.register(coordinator_id)

    # ------------------------------------------------------------------
    def request_exploration(
        self,
        wanderer_id: str,
        *,
        seed: int,
        max_steps: int,
        device: str | None = None,
        timeout: float | None = None,
    ) -> ExplorationResult:
        """Send an exploration request and wait for the result."""

        send_exploration_request(
            self._bus,
            self._coordinator_id,
            wanderer_id,
            seed=seed,
            max_steps=max_steps,
            device=device,
            delay=self._latency,
        )
        msg = self._bus.receive(self._coordinator_id, timeout=timeout or self._timeout)
        return receive_exploration_result(msg)
