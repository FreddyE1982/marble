"""MARBLEAgent wrapper providing per-agent configuration and lifecycle."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from config_loader import create_marble_from_config, load_config
from message_bus import MessageBus, Message


@dataclass
class MARBLEAgent:
    """High level agent encapsulating its own MARBLE brain and config.

    Parameters
    ----------
    agent_id:
        Unique identifier for the agent on the :class:`MessageBus`.
    config_path:
        Optional path to a YAML configuration overriding the default
        ``config.yaml``.  Each agent can supply its own configuration file.
    bus:
        Shared :class:`MessageBus` used for inter-agent communication.
    overrides:
        Optional dictionary of configuration overrides applied on top of the
        loaded YAML configuration.
    """

    agent_id: str
    config_path: Optional[str] = None
    bus: Optional[MessageBus] = None
    overrides: Optional[dict] = None
    eager: bool = True

    def __post_init__(self) -> None:
        self.config = load_config(self.config_path) if self.config_path else load_config()
        if self.overrides:
            from config_loader import _deep_update  # reuse helper

            _deep_update(self.config, self.overrides)
        self.marble = None
        self.brain = None
        if self.eager:
            self.marble = create_marble_from_config(self.config_path, overrides=self.overrides)
            self.brain = self.marble.brain
        if self.bus:
            self.bus.register(self.agent_id)

    # ------------------------------------------------------------------
    # Communication helpers
    def send(self, recipient: str, content: dict) -> None:
        if not self.bus:
            raise RuntimeError("Agent has no MessageBus")
        self.bus.send(self.agent_id, recipient, content)

    def broadcast(self, content: dict) -> None:
        if not self.bus:
            raise RuntimeError("Agent has no MessageBus")
        self.bus.broadcast(self.agent_id, content)

    def receive(self, timeout: float | None = None) -> Message | None:
        if not self.bus:
            raise RuntimeError("Agent has no MessageBus")
        try:
            return self.bus.receive(self.agent_id, timeout=timeout)
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Lifecycle management
    def start(self) -> None:
        """Start brain auto-fire and dreaming threads if available."""
        if self.brain and hasattr(self.brain, "start_auto_fire"):
            self.brain.start_auto_fire()
        if self.brain and hasattr(self.brain, "start_dreaming"):
            self.brain.start_dreaming()

    def stop(self) -> None:
        """Stop brain threads gracefully."""
        if self.brain and hasattr(self.brain, "stop_auto_fire"):
            self.brain.stop_auto_fire()
        if self.brain and hasattr(self.brain, "stop_dreaming"):
            self.brain.stop_dreaming()

    def act(self, observation) -> int:
        """Dummy policy mapping observation to an action.

        Real integrations should delegate to the underlying ``marble`` brain,
        but for testing we return a constant action.
        """
        return 0

    def on_reward(self, reward: float) -> None:  # pragma: no cover - placeholder
        """Hook invoked with received reward. Can be overridden by subclasses."""
        pass

