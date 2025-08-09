"""Thread-safe messaging infrastructure for multi-agent MARBLE."""

import time
from dataclasses import dataclass
from queue import Queue, Empty
from threading import Event, Lock, Thread
from typing import Callable, Dict, List, Optional

import networkx as nx
from event_bus import global_event_bus

__all__ = ["Message", "MessageBus", "AsyncDispatcher"]

@dataclass
class Message:
    """Represents a single message exchanged between agents."""

    sender: str
    recipient: Optional[str]
    content: dict
    timestamp: float


class MessageBus:
    """Thread-safe message bus supporting direct and broadcast messages.

    Each registered agent has a dedicated :class:`queue.Queue` instance to
    receive messages.  Messages are recorded for later inspection and can be
    converted into a NetworkX graph for dashboard visualisation.
    """

    def __init__(self) -> None:
        self._queues: Dict[str, Queue] = {}
        self._lock = Lock()
        self._history: List[Message] = []

    # ------------------------------------------------------------------
    # Registration
    def register(self, agent_id: str) -> None:
        """Register a new agent with an empty message queue."""
        with self._lock:
            if agent_id not in self._queues:
                self._queues[agent_id] = Queue()

    # ------------------------------------------------------------------
    # Sending
    def send(self, sender: str, recipient: str, content: dict) -> None:
        """Send a direct message from ``sender`` to ``recipient``."""
        msg = Message(sender, recipient, content, time.time())
        with self._lock:
            if recipient not in self._queues:
                raise KeyError(f"Unknown recipient '{recipient}'")
            self._queues[recipient].put(msg)
            self._history.append(msg)
        global_event_bus.publish("agent_message", msg.__dict__)

    def broadcast(self, sender: str, content: dict) -> None:
        """Broadcast a message to all agents except the sender."""
        ts = time.time()
        with self._lock:
            for agent_id, q in self._queues.items():
                if agent_id == sender:
                    continue
                msg = Message(sender, agent_id, content, ts)
                q.put(msg)
                self._history.append(msg)
                global_event_bus.publish("agent_message", msg.__dict__)
    def reply(self, original: Message, content: dict) -> None:
        """Send a direct reply to ``original`` sender.

        Parameters
        ----------
        original:
            The :class:`Message` being responded to. The reply will be queued
            for ``original.sender`` and will appear as sent from
            ``original.recipient``.
        content:
            Payload dictionary for the reply message.
        """
        recipient = original.sender
        if recipient is None:
            raise KeyError("Original message has no sender to reply to")
        msg = Message(original.recipient or "", recipient, content, time.time())
        with self._lock:
            if recipient not in self._queues:
                raise KeyError(f"Unknown recipient '{recipient}'")
            self._queues[recipient].put(msg)
            self._history.append(msg)
        global_event_bus.publish("agent_message", msg.__dict__)
    # ------------------------------------------------------------------
    # Receiving
    def receive(self, agent_id: str, timeout: Optional[float] = None) -> Message:
        """Retrieve the next message for ``agent_id``.

        Raises
        ------
        queue.Empty
            If no message arrives within ``timeout`` seconds.
        """
        if agent_id not in self._queues:
            raise KeyError(f"Unknown agent '{agent_id}'")
        return self._queues[agent_id].get(timeout=timeout)

    # ------------------------------------------------------------------
    # History and metrics
    @property
    def history(self) -> List[Message]:
        """Return list of all messages exchanged so far."""
        return list(self._history)

    def influence_graph(self) -> nx.DiGraph:
        """Return a directed graph representing communication flows."""
        g = nx.DiGraph()
        for msg in self._history:
            if g.has_edge(msg.sender, msg.recipient):
                g[msg.sender][msg.recipient]["weight"] += 1
            else:
                g.add_edge(msg.sender, msg.recipient, weight=1)
        return g


class AsyncDispatcher:
    """Background thread dispatching incoming messages to a handler.

    Parameters
    ----------
    bus:
        The :class:`MessageBus` instance to listen on.
    agent_id:
        Identifier of the agent whose queue should be monitored.
    handler:
        Callback invoked with each :class:`Message` received.
    poll_interval:
        Time in seconds to wait when polling the queue before checking the
        stop flag.
    """

    def __init__(
        self,
        bus: MessageBus,
        agent_id: str,
        handler: Callable[[Message], None],
        *,
        poll_interval: float = 0.1,
    ) -> None:
        self._bus = bus
        self._agent_id = agent_id
        self._handler = handler
        self._poll_interval = poll_interval
        self._stop = Event()
        self._thread: Optional[Thread] = None

    def start(self) -> None:
        """Start dispatching messages in a background thread."""
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                msg = self._bus.receive(self._agent_id, timeout=self._poll_interval)
            except Empty:
                continue
            self._handler(msg)

    def stop(self) -> None:
        """Stop the background dispatcher and wait for it to terminate."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join()

