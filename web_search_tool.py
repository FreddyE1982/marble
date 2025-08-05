from __future__ import annotations

"""Tool plugin performing web searches using the DuckDuckGo API."""

from typing import Any

import requests
import torch

from tool_plugins import ToolPlugin, register_tool


class WebSearchTool(ToolPlugin):
    """Perform simple web searches via the DuckDuckGo API."""

    def can_handle(self, query: str) -> bool:
        q = query.lower()
        return any(k in q for k in ("search", "web", "internet"))

    def execute(self, device: torch.device, marble=None, query: str = "") -> Any:
        resp = requests.get(
            "https://api.duckduckgo.com/",
            params={"q": query, "format": "json"},
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json()


def register(register_fn=register_tool) -> None:
    """Entry point used by :func:`load_tool_plugins`."""

    register_fn("web_search", WebSearchTool)
