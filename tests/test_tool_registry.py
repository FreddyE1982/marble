from __future__ import annotations

import torch

from tool_plugins import ToolPlugin, get_tool, register_tool, TOOL_REGISTRY


class _EchoTool(ToolPlugin):
    def can_handle(self, query: str) -> bool:  # pragma: no cover - trivial
        return True

    def execute(self, device: torch.device, marble=None, query: str = ""):
        return query


def test_register_and_get_tool() -> None:
    register_tool("echo", _EchoTool)
    assert get_tool("echo") is _EchoTool
    assert "echo" in TOOL_REGISTRY
