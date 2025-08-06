from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import requests

from dataset_loader import export_dataset, load_dataset
from dataset_versioning import apply_version, create_version


def list_versions(registry: str) -> List[str]:
    """Return available version identifiers from ``registry``."""
    if registry.startswith("http://") or registry.startswith("https://"):
        url = registry.rstrip("/") + "/index.json"
        resp = requests.get(url, timeout=5)
        resp.raise_for_status()
        return resp.json()
    path = Path(registry)
    if not path.exists():
        return []
    return [p.stem for p in path.glob("*.json")]


def create_version_cmd(base: str, new: str, registry: str) -> str:
    """Create a dataset diff between ``base`` and ``new`` in ``registry``."""
    base_pairs = _to_python_pairs(load_dataset(base))
    new_pairs = _to_python_pairs(load_dataset(new))
    return create_version(base_pairs, new_pairs, registry)


def switch_version(base: str, version_id: str, registry: str, output: str) -> None:
    """Apply ``version_id`` from ``registry`` to ``base`` and save to ``output``."""
    pairs = load_dataset(base)
    updated = apply_version(_to_python_pairs(pairs), registry, version_id)
    export_dataset(updated, output)


def _to_python_pairs(pairs: list[tuple[object, object]]) -> list[tuple[object, object]]:
    """Return ``pairs`` with numpy scalars converted to native Python types."""

    out: list[tuple[object, object]] = []
    for a, b in pairs:
        try:
            a = a.item()  # type: ignore[attr-defined]
        except Exception:
            pass
        try:
            b = b.item()  # type: ignore[attr-defined]
        except Exception:
            pass
        out.append((a, b))
    return out


def _main() -> None:
    parser = argparse.ArgumentParser(description="Manage dataset versions")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_list = sub.add_parser("list", help="List available versions")
    p_list.add_argument("--registry", required=True)

    p_create = sub.add_parser("create", help="Create a new version from two datasets")
    p_create.add_argument("base")
    p_create.add_argument("new")
    p_create.add_argument("--registry", required=True)

    p_switch = sub.add_parser("switch", help="Apply a version to a dataset")
    p_switch.add_argument("base")
    p_switch.add_argument("version")
    p_switch.add_argument("output")
    p_switch.add_argument("--registry", required=True)

    args = parser.parse_args()

    if args.cmd == "list":
        versions = list_versions(args.registry)
        print("\n".join(versions))
    elif args.cmd == "create":
        vid = create_version_cmd(args.base, args.new, args.registry)
        print(vid)
    elif args.cmd == "switch":
        switch_version(args.base, args.version, args.registry, args.output)
    else:  # pragma: no cover - defensive
        parser.error("Unknown command")


if __name__ == "__main__":  # pragma: no cover - CLI entry
    _main()
