"""Thread-safe messaging infrastructure for multi-agent MARBLE."""

import time
from dataclasses import dataclass
from queue import Queue, Empty
from threading import Lock
from typing import Dict, List, Optional

import networkx as nx
from event_bus import global_event_bus


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

