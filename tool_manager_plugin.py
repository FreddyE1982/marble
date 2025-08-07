from __future__ import annotations

"""Pipeline plugin that routes queries to external tools."""

from typing import Dict, Optional

import torch

from pipeline_plugins import PipelinePlugin, register_plugin
from tool_plugins import (
    TOOL_REGISTRY,
    ToolPlugin,
    get_tool,
    load_tool_plugins,
    register_tool,
)
from message_bus import MessageBus
from queue import Empty
import threading


class HeuristicToolPolicy:
    """Simple policy selecting the first tool able to handle a query."""

    def select(self, query: str, tools: Dict[str, ToolPlugin]) -> str:
        for name, tool in tools.items():
            try:
                if tool.can_handle(query):
                    return name
            except Exception:
                continue
        return next(iter(tools))


class ToolManagerPlugin(PipelinePlugin):
    """Meta-plugin that delegates a query to one of several tools."""

    def __init__(
        self,
        tools: Dict[str, Dict],
        policy: str = "heuristic",
        *,
        mode: str = "direct",
        bus: Optional[MessageBus] = None,
        agent_id: str = "tool_manager",
    ) -> None:
        super().__init__(tools=tools, policy=policy, mode=mode, agent_id=agent_id)
        self.tool_configs = tools
        self.policy_name = policy
        self.mode = mode
        self.bus = bus
        self.agent_id = agent_id
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._device: Optional[torch.device] = None
        self._marble = None

    def initialise(self, device: torch.device, marble=None) -> None:
        load_tool_plugins()
        self._tools: Dict[str, ToolPlugin] = {}
        for name, cfg in self.tool_configs.items():
            if name not in TOOL_REGISTRY:
                try:
                    mod = __import__(f"{name}_tool")
                    if hasattr(mod, "register"):
                        mod.register(register_tool)
                except ImportError as exc:  # pragma: no cover - user error
                    raise ImportError(f"Tool '{name}' is not registered") from exc
            cls = get_tool(name)
            inst = cls(**cfg)
            inst.initialise(device, marble)
            self._tools[name] = inst
        if self.policy_name == "heuristic":
            self._policy = HeuristicToolPolicy()
        else:  # pragma: no cover - future policies
            raise ValueError(f"Unknown policy {self.policy_name}")
        self._device = device
        self._marble = marble
        if self.mode == "mcp":
            if self.bus is None:
                raise ValueError("MessageBus required in mcp mode")
            self.bus.register(self.agent_id)

    def _run_tool(self, query: str):
        tool_name = self._policy.select(query, self._tools)
        result = self._tools[tool_name].execute(
            self._device if self._device is not None else torch.device("cpu"),
            marble=self._marble,
            query=query,
        )
        return tool_name, result

    def _bus_loop(self) -> None:
        assert self.bus is not None
        while self._running:
            try:
                msg = self.bus.receive(self.agent_id, timeout=0.1)
            except Empty:
                continue
            query = msg.content.get("query", "")
            tool_name, result = self._run_tool(query)
            self.bus.reply(
                msg,
                {
                    "request_id": msg.content.get("request_id"),
                    "tool": tool_name,
                    "result": result,
                },
            )

    def execute(self, device: torch.device, marble=None, query: str = ""):
        if self.mode == "mcp":
            if not self._running:
                self._running = True
                self._thread = threading.Thread(target=self._bus_loop, daemon=True)
                self._thread.start()
            return {"mode": "mcp", "agent_id": self.agent_id}
        if not query:
            raise ValueError("query must be supplied")
        tool_name, result = self._run_tool(query)
        return {"tool": tool_name, "result": result}

    def teardown(self) -> None:
        for tool in self._tools.values():
            tool.teardown()
        if self.mode == "mcp" and self._thread is not None:
            self._running = False
            self._thread.join(timeout=1)
            self._thread = None


def register(register_fn=register_plugin) -> None:
    """Entry point used by :func:`load_pipeline_plugins`."""

    register_fn("tool_manager", ToolManagerPlugin)
