"""Utilities for synchronising configuration files across nodes."""

from __future__ import annotations

import shutil
from pathlib import Path


def sync_config(src: str, dest_paths: list[str]) -> None:
    """Copy ``src`` config file to all ``dest_paths``."""
    source = Path(src)
    if not source.is_file():
        raise FileNotFoundError(src)
    for dest in dest_paths:
        dest_path = Path(dest)
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, dest_path)
