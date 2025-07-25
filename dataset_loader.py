import os
import hashlib
import requests
import pandas as pd
from typing import Any


def _download_file(url: str, path: str) -> None:
    """Download ``url`` to ``path`` creating parent directories if needed."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with requests.get(url, stream=True, timeout=30) as r:
        r.raise_for_status()
        with open(path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)


def load_dataset(
    source: str,
    *,
    cache_dir: str = "dataset_cache",
    input_col: str = "input",
    target_col: str = "target",
    limit: int | None = None,
    force_refresh: bool = False,
) -> list[tuple[Any, Any]]:
    """Load a CSV or JSON dataset from ``source`` which may be a local path or URL."""
    if source.startswith("http://") or source.startswith("https://"):
        name = os.path.basename(source)
        if not name:
            name = hashlib.md5(source.encode("utf-8")).hexdigest() + ".dat"
        cached = os.path.join(cache_dir, name)
        if force_refresh or not os.path.exists(cached):
            _download_file(source, cached)
        path = cached
    else:
        path = source
    ext = os.path.splitext(path)[1].lower()
    if ext in {".csv", ""}:
        df = pd.read_csv(path)
    elif ext in {".json", ".jsonl"}:
        df = pd.read_json(path, lines=ext == ".jsonl")
    else:
        raise ValueError("Unsupported dataset format")
    pairs: list[tuple[Any, Any]] = []
    for _, row in df.iterrows():
        pairs.append((row[input_col], row[target_col]))
        if limit is not None and len(pairs) >= limit:
            break
    return pairs
