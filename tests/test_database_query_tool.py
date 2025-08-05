from __future__ import annotations

from unittest.mock import Mock, patch

import torch

from database_query_tool import DatabaseQueryTool


def test_database_query_can_handle() -> None:
    tool = DatabaseQueryTool(db_path="db.kuzu")
    assert tool.can_handle("database query for data")
    assert not tool.can_handle("web search request")


@patch("database_query_tool.KuzuGraphDatabase")
def test_database_query_execute(mock_db_cls: Mock) -> None:
    mock_db = Mock()
    mock_db.execute.return_value = [{"x": 1}]
    mock_db_cls.return_value = mock_db

    tool = DatabaseQueryTool(db_path="db.kuzu")
    tool.initialise(torch.device("cpu"))
    result = tool.execute(torch.device("cpu"), query="MATCH RETURN 1")
    mock_db.execute.assert_called_once_with("MATCH RETURN 1")
    assert result == [{"x": 1}]
    tool.teardown()
    mock_db.close.assert_called_once()
