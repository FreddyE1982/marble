import asyncio
import time
from unittest.mock import patch

import aiohttp
import torch

from message_bus import MessageBus
from tool_manager_plugin import ToolManagerPlugin
from tool_plugins import register_tool
from web_search_tool import WebSearchTool
from mcp_tool_bridge import MCPToolBridge


def test_mcp_tool_bridge_roundtrip():
    bus = MessageBus()
    register_tool("web_search", WebSearchTool)
    with patch("web_search_tool.WebSearchTool.execute", return_value={"ok": 1}):
        manager = ToolManagerPlugin(tools={"web_search": {}}, mode="mcp", bus=bus)
        manager.initialise(torch.device("cpu"))
        manager.execute(torch.device("cpu"))
        bridge = MCPToolBridge(bus, host="localhost", port=5084)
        bridge.start()
        time.sleep(0.2)
        try:
            async def _request():
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        "http://localhost:5084/mcp/tool", json={"query": "search"}
                    ) as resp:
                        assert resp.status == 200
                        data = await resp.json()
                        assert data["tool"] == "web_search"
                        assert data["result"] == {"ok": 1}
            asyncio.run(_request())
        finally:
            bridge.stop()
            manager.teardown()
