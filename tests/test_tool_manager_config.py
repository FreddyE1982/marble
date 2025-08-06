import yaml
from pathlib import Path
import torch

from config_loader import create_marble_from_config, load_config


def test_tool_manager_loaded_from_config(tmp_path, monkeypatch):
    cfg = load_config()
    cfg["tool_manager"] = {
        "enabled": True,
        "policy": "heuristic",
        "tools": {
            "web_search": {},
            "database_query": {"db_path": str(tmp_path / "db.kuzu")},
        },
    }
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))

    called = {}

    def fake_ws(self, device, marble=None, query=""):
        called["web"] = True
        return {"ok": 1}

    def fake_db(self, device, marble=None, query=""):
        called["db"] = True
        return [{"x": 1}]

    monkeypatch.setattr("web_search_tool.WebSearchTool.execute", fake_ws)
    monkeypatch.setattr("database_query_tool.DatabaseQueryTool.execute", fake_db)

    marble = create_marble_from_config(str(cfg_path))
    res1 = marble.tool_manager.execute(torch.device("cpu"), query="search the web")
    assert res1["tool"] == "web_search"
    assert called["web"]
    res2 = marble.tool_manager.execute(torch.device("cpu"), query="database stats")
    assert res2["tool"] == "database_query"
    assert called["db"]
    marble.tool_manager.teardown()
