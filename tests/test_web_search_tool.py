from __future__ import annotations

from unittest.mock import Mock, patch

import torch

from web_search_tool import WebSearchTool


def test_web_search_can_handle() -> None:
    tool = WebSearchTool()
    assert tool.can_handle("search the web for cats")
    assert not tool.can_handle("database lookup")


@patch("web_search_tool.requests.get")
def test_web_search_execute(mock_get: Mock) -> None:
    mock_resp = Mock()
    mock_resp.json.return_value = {"Abstract": "Cats"}
    mock_resp.raise_for_status.return_value = None
    mock_get.return_value = mock_resp

    tool = WebSearchTool()
    result = tool.execute(torch.device("cpu"), query="cats")
    mock_get.assert_called_once()
    assert result == {"Abstract": "Cats"}
