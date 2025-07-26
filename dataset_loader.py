import os
import hashlib
import zipfile
import io
import requests
import pandas as pd
from typing import Any
from tqdm import tqdm


def _download_file(url: str, path: str) -> None:
    """Download ``url`` to ``path`` creating parent directories if needed."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with requests.get(url, stream=True, timeout=30) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        desc = f"Downloading {os.path.basename(path)}"
        with open(path, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, desc=desc
        ) as pbar:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))


def load_dataset(
    source: str,
    *,
    cache_dir: str = "dataset_cache",
    input_col: str = "input",
    target_col: str = "target",
    limit: int | None = None,
    force_refresh: bool = False,
) -> list[tuple[Any, Any]]:
    """Load a dataset from ``source``.

    The ``source`` can be a local path, remote URL, or a ZIP archive containing
    either a CSV or JSON/JSONL file. Remote sources are automatically cached in
    ``cache_dir`` and reused on subsequent calls unless ``force_refresh`` is
    ``True``. If the dataset is zipped only the first CSV/JSON file inside the
    archive is used.
    """
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
    if ext == ".zip":
        with zipfile.ZipFile(path) as zf:
            members = [n for n in zf.namelist() if not n.endswith("/")]
            if not members:
                raise ValueError("Zip archive is empty")
            inner = members[0]
            with zf.open(inner) as f:
                inner_ext = os.path.splitext(inner)[1].lower()
                if inner_ext in {".csv", ""}:
                    df = pd.read_csv(f)
                elif inner_ext in {".json", ".jsonl"}:
                    df = pd.read_json(f, lines=inner_ext == ".jsonl")
                else:
                    raise ValueError("Unsupported dataset format inside zip")
    elif ext in {".csv", ""}:
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
