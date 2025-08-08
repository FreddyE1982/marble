from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
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


def list_config_keys(cfg_path: str | Path) -> List[str]:
    """Return dot-separated keys defined in the YAML configuration."""
    with open(cfg_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return sorted(_collect_keys(data))


def find_unused_keys(cfg_path: str | Path, root: str | Path = ".") -> List[str]:
    """Return configuration keys that are not referenced in the codebase.

    Parameters
    ----------
    cfg_path:
        Path to the YAML configuration file to inspect.
    root:
        Directory that will be scanned recursively for usages. Only Python
        source files are considered to reduce false positives.
    """

    keys = list_config_keys(cfg_path)
    root_path = Path(root)

    pattern_map: dict[str, str] = {}
    for key in keys:
        parts = key.split(".")
        pattern_map[key] = key
        pattern_map["".join(f"['{p}']" for p in parts)] = key
        pattern_map["".join(f'["{p}"]' for p in parts)] = key

    import json
    import tempfile

    with tempfile.NamedTemporaryFile("w", delete=False) as tmp:
        for pat in pattern_map.keys():
            tmp.write(pat + "\n")
        pattern_file = tmp.name

    try:
        try:
            result = subprocess.run(
                [
                    "rg",
                    "-t",
                    "py",
                    "-F",
                    "-f",
                    pattern_file,
                    str(root_path),
                    "-g",
                    "!config.yaml",
                    "--json",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
        except FileNotFoundError as exc:  # pragma: no cover - requires rg
            raise RuntimeError("ripgrep (rg) is required for usage scanning") from exc
    finally:
        Path(pattern_file).unlink(missing_ok=True)

    found: set[str] = set()
    for line in result.stdout.splitlines():
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue
        if data.get("type") == "match":
            for sub in data["data"].get("submatches", []):
                text = sub.get("match", {}).get("text")
                key = pattern_map.get(text)
                if key:
                    found.add(key)

    return [k for k in keys if k not in found]


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect configuration keys")
    parser.add_argument(
        "config", nargs="?", default="config.yaml", help="Path to YAML config"
    )
    parser.add_argument(
        "--root",
        default=".",
        help="Project root to scan for key usages when --unused is set",
    )
    parser.add_argument(
        "--unused",
        action="store_true",
        help="List keys that are not referenced in Python sources",
    )
    parser.add_argument(
        "--report",
        help="Optional path to write the list of unused keys",
    )

    args = parser.parse_args()
    if args.unused:
        unused = find_unused_keys(args.config, args.root)
        if args.report:
            Path(args.report).write_text("\n".join(unused), encoding="utf-8")
        for key in unused:
            print(key)
    else:
        for key in list_config_keys(args.config):
            print(key)


if __name__ == "__main__":
    main()
