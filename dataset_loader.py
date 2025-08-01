import os
import hashlib
import zipfile
import pickle
import requests
import torch.distributed as dist
import requests_cache
import pandas as pd
import threading
from typing import Any, List
from marble import DataLoader

_SESSION = requests_cache.CachedSession("http_cache", expire_after=86400)
from tqdm import tqdm


def distributed_shard(pairs: list[tuple[Any, Any]]) -> list[tuple[Any, Any]]:
    """Return subset of ``pairs`` for the current distributed rank."""
    if dist.is_available() and dist.is_initialized():
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        return pairs[rank::world_size]
    return pairs


def _hash_value(val: Any, dl: DataLoader | None) -> str:
    """Return SHA256 hash for ``val`` using ``dl`` if provided."""
    if dl is not None:
        tensor = dl.encode(val)
        try:
            data = tensor.tobytes()
        except Exception:
            data = pickle.dumps(tensor)
    else:
        data = pickle.dumps(val)
    return hashlib.sha256(data).hexdigest()


def _download_file(url: str, path: str) -> None:
    """Download ``url`` to ``path`` creating parent directories if needed."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with _SESSION.get(url, stream=True, timeout=30) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        desc = f"Downloading {os.path.basename(path)}"
        with open(path, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, desc=desc
        ) as pbar:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))


_PREFETCH_THREADS: List[threading.Thread] = []


def prefetch_dataset(
    source: str, *, cache_dir: str = "dataset_cache", force_refresh: bool = False
) -> threading.Thread:
    """Prefetch ``source`` to the cache in a background thread.

    Parameters
    ----------
    source:
        Dataset URL to download. Local paths return an already completed thread.
    cache_dir:
        Directory where the downloaded file will be cached.
    force_refresh:
        When ``True`` download even if the file is already cached.

    Returns
    -------
    threading.Thread
        Thread handle performing the download. Join the thread to wait.
    """

    def _noop() -> None:
        pass

    if not (source.startswith("http://") or source.startswith("https://")):
        t = threading.Thread(target=_noop, daemon=True)
        t.start()
        return t

    name = os.path.basename(source)
    if not name:
        name = hashlib.md5(source.encode("utf-8")).hexdigest() + ".dat"
    path = os.path.join(cache_dir, name)
    if not force_refresh and os.path.exists(path):
        t = threading.Thread(target=_noop, daemon=True)
        t.start()
        return t

    def _task() -> None:
        _download_file(source, path)

    thread = threading.Thread(target=_task, daemon=True)
    thread.start()
    _PREFETCH_THREADS.append(thread)
    return thread


def wait_for_prefetch() -> None:
    """Block until all outstanding prefetch threads finished."""

    while _PREFETCH_THREADS:
        t = _PREFETCH_THREADS.pop(0)
        t.join()


def load_dataset(
    source: str,
    *,
    cache_dir: str = "dataset_cache",
    input_col: str = "input",
    target_col: str = "target",
    limit: int | None = None,
    force_refresh: bool = False,
    offline: bool = False,
    num_shards: int | None = None,
    shard_index: int = 0,
    dataloader: "DataLoader | None" = None,
    return_deps: bool = False,
    filter_expr: str | None = None,
) -> list[tuple[Any, Any]] | tuple[list[tuple[Any, Any]], list[dict]]:
    """Load a dataset from ``source``.

    The ``source`` can be a local path, remote URL, or a ZIP archive containing
    either a CSV or JSON/JSONL file. Remote sources are automatically cached in
    ``cache_dir`` and reused on subsequent calls unless ``force_refresh`` is
    ``True``. When ``offline`` is ``True`` the function will only load from the
    local cache and raise ``FileNotFoundError`` if the file is not present. If
    the dataset is zipped only the first CSV/JSON file inside the
    archive is used. Large datasets can be split across multiple shards by
    specifying ``num_shards`` and ``shard_index``. When ``num_shards`` is
    greater than one, only every ``num_shards``-th sample starting at
    ``shard_index`` is returned. This is useful for distributed training where
    each worker processes a different shard. When ``dataloader`` is provided,
    inputs and targets are encoded using :class:`~marble.DataLoader`.
    """
    if source.startswith("http://") or source.startswith("https://"):
        name = os.path.basename(source)
        if not name:
            name = hashlib.md5(source.encode("utf-8")).hexdigest() + ".dat"
        cached = os.path.join(cache_dir, name)
        if offline:
            if not os.path.exists(cached):
                raise FileNotFoundError(
                    f"{cached} not available in offline mode"
                )
        elif force_refresh or not os.path.exists(cached):
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
    deps: list[dict] = []
    for _, row in df.iterrows():
        inp = row[input_col]
        tgt = row[target_col]
        inp_src = inp
        tgt_src = tgt
        for val_name, val in [("input", inp), ("target", tgt)]:
            if isinstance(val, str) and val.startswith(("http://", "https://")):
                name = os.path.basename(val)
                if not name:
                    name = hashlib.md5(val.encode("utf-8")).hexdigest()
                cached_path = os.path.join(cache_dir, name)
                if not os.path.exists(cached_path):
                    _download_file(val, cached_path)
                with open(cached_path, "rb") as f:
                    data_bytes = f.read()
                if val_name == "input":
                    inp = data_bytes
                    inp_src = val
                else:
                    tgt = data_bytes
                    tgt_src = val
        if dataloader is not None:
            inp = dataloader.encode(inp)
            tgt = dataloader.encode(tgt)
        if filter_expr is None or eval(
            filter_expr,
            {"__builtins__": {}},
            {"input": inp if dataloader is None else dataloader.decode(inp), "target": tgt if dataloader is None else dataloader.decode(tgt)},
        ):
            pairs.append((inp, tgt))
            deps.append(
                {
                    "id": _hash_value((inp, tgt), None),
                    "input_source": inp_src,
                    "target_source": tgt_src,
                }
            )
            if limit is not None and len(pairs) >= limit:
                break

    if num_shards and num_shards > 1:
        if shard_index < 0 or shard_index >= num_shards:
            raise ValueError("shard_index must be within [0, num_shards)")
        pairs = pairs[shard_index::num_shards]
    elif num_shards is None:
        pairs = distributed_shard(pairs)

    if return_deps:
        return pairs, deps
    return pairs


def export_dataset(pairs: list[tuple[Any, Any]], path: str) -> None:
    """Save ``pairs`` to ``path`` in CSV or JSON format."""
    df = pd.DataFrame(pairs, columns=["input", "target"])
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv" or not ext:
        df.to_csv(path, index=False)
    elif ext in {".json", ".jsonl"}:
        df.to_json(path, orient="records", lines=ext == ".jsonl")
    else:
        raise ValueError("Unsupported export format")
