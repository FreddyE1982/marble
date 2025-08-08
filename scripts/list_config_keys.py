from __future__ import annotations

import argparse
import pathlib
from typing import Any, List

import yaml


def _collect_keys(obj: Any, prefix: str = "") -> List[str]:
    keys: List[str] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            path = f"{prefix}{k}" if not prefix else f"{prefix}.{k}"
            keys.append(path)
            keys.extend(_collect_keys(v, path))
    return keys


def list_config_keys(cfg_path: str | pathlib.Path) -> List[str]:
    """Return dot-separated keys defined in the YAML configuration."""
    with open(cfg_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return sorted(_collect_keys(data))


def main() -> None:
    parser = argparse.ArgumentParser(description="List configuration keys")
    parser.add_argument(
        "config", nargs="?", default="config.yaml", help="Path to YAML config"
    )
    args = parser.parse_args()
    for key in list_config_keys(args.config):
        print(key)


if __name__ == "__main__":
    main()
