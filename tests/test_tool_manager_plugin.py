from __future__ import annotations

from unittest.mock import patch

import torch

from tool_manager_plugin import ToolManagerPlugin
from tool_plugins import register_tool
from web_search_tool import WebSearchTool
from database_query_tool import DatabaseQueryTool


# Ensure tools are registered for the manager
register_tool("web_search", WebSearchTool)
register_tool("database_query", DatabaseQueryTool)


def test_manager_selects_web_search() -> None:
    with patch("web_search_tool.WebSearchTool.execute", return_value={"ok": 1}) as mock_ws:
        with patch("database_query_tool.DatabaseQueryTool.execute", return_value=[{"x": 1}]):
            manager = ToolManagerPlugin(
                tools={"web_search": {}, "database_query": {"db_path": "db.kuzu"}}
            )
            manager.initialise(torch.device("cpu"))
            result = manager.execute(torch.device("cpu"), query="search the web")
            assert result["tool"] == "web_search"
            mock_ws.assert_called_once()
            manager.teardown()


def test_manager_selects_database() -> None:
    with patch("database_query_tool.DatabaseQueryTool.execute", return_value=[{"x": 1}]) as mock_db:
        with patch("web_search_tool.WebSearchTool.execute", return_value={"ok": 1}):
            manager = ToolManagerPlugin(
                tools={"web_search": {}, "database_query": {"db_path": "db.kuzu"}}
            )
            manager.initialise(torch.device("cpu"))
            result = manager.execute(torch.device("cpu"), query="database stats")
            assert result["tool"] == "database_query"
            mock_db.assert_called_once()
            manager.teardown()
