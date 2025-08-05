"""High-level interface for interacting with a persistent Kùzu graph database.

This module provides a :class:`KuzuGraphDatabase` class that wraps the
:mod:`kuzu` Python API and exposes convenience methods for common graph
operations. All queries use the Cypher query language supported by Kùzu.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import kuzu


class KuzuGraphDatabase:
    """A convenience wrapper around :mod:`kuzu` for persistent graph storage.

    Parameters
    ----------
    db_path:
        Filesystem path to the database. The path should point to a location
        that does not already exist; a new database will be created there.
        To reopen an existing database simply pass the same path again.
    """

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        self._db = kuzu.Database(db_path)
        self._conn = kuzu.Connection(self._db)

    # ------------------------------------------------------------------
    # context management utilities
    # ------------------------------------------------------------------
    def __enter__(self) -> "KuzuGraphDatabase":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # type: ignore[override]
        self.close()

    def close(self) -> None:
        """Close the underlying connection."""
        if hasattr(self, "_conn") and self._conn is not None:
            self._conn.close()
            self._conn = None

    # ------------------------------------------------------------------
    # schema operations
    # ------------------------------------------------------------------
    def create_node_table(
        self, label: str, columns: Dict[str, str], primary_key: str
    ) -> None:
        """Create a node table with ``label`` and ``columns``.

        ``columns`` maps column names to Kùzu data types. ``primary_key`` must
        be one of the column names.
        """

        cols = ", ".join(f"{name} {dtype}" for name, dtype in columns.items())
        query = f"CREATE NODE TABLE {label}({cols}, PRIMARY KEY({primary_key}));"
        self.execute(query)

    def create_relationship_table(
        self,
        rel_name: str,
        src_table: str,
        dst_table: str,
        columns: Optional[Dict[str, str]] = None,
    ) -> None:
        """Create a relationship table.

        Parameters
        ----------
        rel_name:
            Name of the relationship type.
        src_table:
            Source node table.
        dst_table:
            Destination node table.
        columns:
            Optional mapping of additional relationship properties and their
            data types.
        """

        cols = (
            ", " + ", ".join(f"{k} {v}" for k, v in columns.items()) if columns else ""
        )
        query = f"CREATE REL TABLE {rel_name}(FROM {src_table} TO {dst_table}{cols});"
        self.execute(query)

    # ------------------------------------------------------------------
    # data manipulation helpers
    # ------------------------------------------------------------------
    def add_node(self, label: str, properties: Dict[str, Any]) -> None:
        """Insert a node with ``label`` and ``properties``."""
        placeholders = ", ".join(f"{k}: ${k}" for k in properties)
        query = f"CREATE (:{label} {{{placeholders}}});"
        self.execute(query, properties)

    def add_relationship(
        self,
        src_label: str,
        src_key: str,
        src_value: Any,
        rel_type: str,
        dst_label: str,
        dst_key: str,
        dst_value: Any,
        properties: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Create a relationship between two existing nodes."""

        prop_str = (
            " {" + ", ".join(f"{k}: ${k}" for k in properties) + "}"
            if properties
            else ""
        )
        params = {"src_val": src_value, "dst_val": dst_value}
        if properties:
            params.update(properties)
        query = (
            f"MATCH (a:{src_label} {{{src_key}: $src_val}}), "
            f"(b:{dst_label} {{{dst_key}: $dst_val}}) "
            f"CREATE (a)-[:{rel_type}{prop_str}]->(b);"
        )
        self.execute(query, params)

    def update_node(
        self, label: str, key: str, key_val: Any, updates: Dict[str, Any]
    ) -> None:
        """Update properties of a node matching ``key`` = ``key_val``."""
        set_clause = ", ".join(f"n.{k} = ${k}" for k in updates)
        params = {"key_val": key_val, **updates}
        query = f"MATCH (n:{label} {{{key}: $key_val}}) SET {set_clause};"
        self.execute(query, params)

    def delete_node(self, label: str, key: str, key_val: Any) -> None:
        """Delete a node and all its relationships."""
        query = f"MATCH (n:{label} {{{key}: $val}}) DETACH DELETE n;"
        self.execute(query, {"val": key_val})

    def delete_relationship(
        self,
        src_label: str,
        src_key: str,
        src_val: Any,
        rel_type: str,
        dst_label: str,
        dst_key: str,
        dst_val: Any,
    ) -> None:
        """Delete a relationship between two nodes."""
        params = {"src_val": src_val, "dst_val": dst_val}
        query = (
            f"MATCH (a:{src_label} {{{src_key}: $src_val}})-"
            f"[r:{rel_type}]->(b:{dst_label} {{{dst_key}: $dst_val}}) DELETE r;"
        )
        self.execute(query, params)

    # ------------------------------------------------------------------
    # query interface
    # ------------------------------------------------------------------
    def execute(
        self, query: str, parameters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Execute ``query`` with optional ``parameters`` and return rows.

        Returns a list of dictionaries mapping column names to values.
        """

        result = self._conn.execute(query, parameters or {})
        cols = result.get_column_names()
        return [dict(zip(cols, row)) for row in result]


__all__ = ["KuzuGraphDatabase"]
