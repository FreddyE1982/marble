"""Dataset versioning utilities with reversible diffs."""

from __future__ import annotations

import json
import os
import uuid
from typing import Any, List, Tuple


def _compute_diff(base: List[Tuple[Any, Any]], new: List[Tuple[Any, Any]]):
    """Return a list of operations turning ``base`` into ``new``."""
    ops = []
    base_set = {tuple(map(json.dumps, p)) for p in base}
    new_set = {tuple(map(json.dumps, p)) for p in new}
    for item in base_set - new_set:
        ops.append({"op": "remove", "pair": item})
    for item in new_set - base_set:
        ops.append({"op": "add", "pair": item})
    return ops


def create_version(
    base: List[Tuple[Any, Any]],
    new: List[Tuple[Any, Any]],
    version_dir: str,
) -> str:
    """Store a diff between ``base`` and ``new`` inside ``version_dir``."""
    os.makedirs(version_dir, exist_ok=True)
    ops = _compute_diff(base, new)
    version_id = str(uuid.uuid4())
    with open(
        os.path.join(version_dir, f"{version_id}.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(ops, f)
    return version_id


def apply_version(
    base: List[Tuple[Any, Any]],
    version_dir: str,
    version_id: str,
) -> List[Tuple[Any, Any]]:
    """Return a dataset patched with ``version_id``."""
    path = os.path.join(version_dir, f"{version_id}.json")
    with open(path, "r", encoding="utf-8") as f:
        ops = json.load(f)
    data = [list(p) for p in base]
    encoded = {tuple(map(json.dumps, p)): list(p) for p in data}
    for op in ops:
        pair = tuple(json.loads(x) for x in op["pair"])
        if op["op"] == "remove":
            encoded.pop(tuple(map(json.dumps, pair)), None)
        elif op["op"] == "add":
            encoded[tuple(map(json.dumps, pair))] = list(pair)
    return [tuple(p) for p in encoded.values()]


def revert_version(
    current: List[Tuple[Any, Any]],
    version_dir: str,
    version_id: str,
) -> List[Tuple[Any, Any]]:
    """Undo the changes from ``version_id`` applied to ``current``."""
    path = os.path.join(version_dir, f"{version_id}.json")
    with open(path, "r", encoding="utf-8") as f:
        ops = json.load(f)
    inverse = []
    for op in ops:
        inverse.append(
            {"op": "add" if op["op"] == "remove" else "remove", "pair": op["pair"]}
        )
    encoded = {tuple(map(json.dumps, p)): list(p) for p in current}
    for op in inverse:
        pair = tuple(json.loads(x) for x in op["pair"])
        if op["op"] == "remove":
            encoded.pop(tuple(map(json.dumps, pair)), None)
        else:
            encoded[tuple(map(json.dumps, pair))] = list(pair)
    return [tuple(p) for p in encoded.values()]
