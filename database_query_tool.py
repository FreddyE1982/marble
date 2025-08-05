from __future__ import annotations

"""Tool plugin executing Cypher queries on a Kùzu database."""

from typing import Any, Dict

import torch

from kuzu_interface import KuzuGraphDatabase
from tool_plugins import ToolPlugin, register_tool


class DatabaseQueryTool(ToolPlugin):
    """Run Cypher queries against a persistent Kùzu database."""

    def __init__(self, db_path: str, **kwargs: Dict[str, Any]) -> None:
        super().__init__(db_path=db_path, **kwargs)
        self.db_path = db_path

    def can_handle(self, query: str) -> bool:
        q = query.lower()
        return any(k in q for k in ("database", "db"))

    def initialise(self, device: torch.device, marble=None) -> None:
        self._db = KuzuGraphDatabase(self.db_path)

    def execute(self, device: torch.device, marble=None, query: str = "") -> Any:
        return self._db.execute(query)

    def teardown(self) -> None:
        if hasattr(self, "_db"):
            self._db.close()


def register(register_fn=register_tool) -> None:
    """Entry point used by :func:`load_tool_plugins`."""

    register_fn("database_query", DatabaseQueryTool)
