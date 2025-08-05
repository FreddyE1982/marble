from __future__ import annotations

"""Pipeline plugin that routes queries to external tools."""

from typing import Dict

import torch

from pipeline_plugins import PipelinePlugin, register_plugin
from tool_plugins import TOOL_REGISTRY, ToolPlugin, get_tool, load_tool_plugins


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

    def __init__(self, tools: Dict[str, Dict], policy: str = "heuristic") -> None:
        super().__init__(tools=tools, policy=policy)
        self.tool_configs = tools
        self.policy_name = policy

    def initialise(self, device: torch.device, marble=None) -> None:
        load_tool_plugins()
        self._tools: Dict[str, ToolPlugin] = {}
        for name, cfg in self.tool_configs.items():
            if name not in TOOL_REGISTRY:
                try:
                    __import__(f"{name}_tool")
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

    def execute(self, device: torch.device, marble=None, query: str = ""):
        if not query:
            raise ValueError("query must be supplied")
        tool_name = self._policy.select(query, self._tools)
        result = self._tools[tool_name].execute(
            device, marble=marble, query=query
        )
        return {"tool": tool_name, "result": result}

    def teardown(self) -> None:
        for tool in self._tools.values():
            tool.teardown()


def register(register_fn=register_plugin) -> None:
    """Entry point used by :func:`load_pipeline_plugins`."""

    register_fn("tool_manager", ToolManagerPlugin)
