import csv
import hashlib
import json
import os
import pickle
import sys
import threading
import zipfile
from typing import Any, List
import io

import pandas as pd
import requests_cache
import torch.distributed as dist
from tqdm import tqdm

from event_bus import global_event_bus
from kuzu_interface import KuzuGraphDatabase
from marble import DataLoader
from marble_base import MetricsVisualizer
from memory_manager import MemoryManager
from memory_pool import MemoryPool
from tokenizer_utils import tokenize_line
from crypto_utils import encrypt_bytes, decrypt_bytes

_SESSION = requests_cache.CachedSession("http_cache", expire_after=86400)
_DATASET_CACHE: dict[str, list[tuple[Any, Any]]] = {}
_DATASET_POOL = MemoryPool(list, max_size=32)


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
        with (
            open(path, "wb") as f,
            tqdm(total=total, unit="B", unit_scale=True, desc=desc) as pbar,
        ):
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))


_PREFETCH_THREADS: List[threading.Thread] = []


def prefetch_dataset(
    source: str,
    *,
    cache_dir: str = "dataset_cache",
    force_refresh: bool = False,
    encryption_key: str | bytes | None = None,
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
    encryption_key:
        Optional key used to XOR-encrypt the cached file. When provided the
        downloaded bytes are encrypted on disk so subsequent loads require the
        same key for decryption.

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
        if encryption_key is not None:
            with open(path, "rb") as f:
                data = f.read()
            with open(path, "wb") as f:
                f.write(b"ENC" + encrypt_bytes(data, encryption_key))

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
    use_cache: bool = True,
    cache_key: str | None = None,
    memory_pool: MemoryPool | None = None,
    memory_manager: "MemoryManager | None" = None,
    metrics_visualizer: "MetricsVisualizer | None" = None,
    filter_expr: str | None = None,
    cache_server_url: str | None = None,
    encryption_key: str | bytes | None = None,
) -> list[tuple[Any, Any]] | tuple[list[tuple[Any, Any]], list[dict]]:
    """Load a dataset from ``source``.

    The ``source`` can be a local path, remote URL, or a ZIP archive containing
    either a CSV or JSON/JSONL file. Remote sources are automatically cached in
    ``cache_dir`` and reused on subsequent calls unless ``force_refresh`` is
    ``True``. When ``offline`` is ``True`` the function will only load from the
    local cache and raise ``FileNotFoundError`` if the file is not present. If
    the dataset is zipped only the first CSV/JSON file inside the archive is
    used. Large datasets can be split across multiple shards by specifying
    ``num_shards`` and ``shard_index``. When ``num_shards`` is greater than one,
    only every ``num_shards``-th sample starting at ``shard_index`` is returned.
    This is useful for distributed training where each worker processes a
    different shard. When ``dataloader`` is provided, inputs and targets are
    encoded using :class:`~marble.DataLoader`. When ``cache_server_url`` is set
    and ``source`` is remote the loader will attempt to fetch the file from the
    cache server before downloading it directly. If ``encryption_key`` is
    provided, cached files are XOR-encrypted with the key and transparently
    decrypted when loading. If ``memory_manager`` is provided, the estimated
    allocated bytes are reported after loading completes.
    """
    if metrics_visualizer:
        metrics_visualizer.log_event("dataset_load_start", {"source": source})
    global_event_bus.publish("dataset_load_start", {"source": source})
    if cache_key is None:
        cache_key = f"{source}:{limit}:{input_col}:{target_col}"

    if offline and (source.startswith("http://") or source.startswith("https://")):
        name = (
            os.path.basename(source)
            or hashlib.md5(source.encode("utf-8")).hexdigest() + ".dat"
        )
        cached = os.path.join(cache_dir, name)
        if not os.path.exists(cached):
            raise FileNotFoundError(f"{cached} not available in offline mode")
    if use_cache and cache_key in _DATASET_CACHE:
        return list(_DATASET_CACHE[cache_key])

    if source.startswith("http://") or source.startswith("https://"):
        name = os.path.basename(source)
        if not name:
            name = hashlib.md5(source.encode("utf-8")).hexdigest() + ".dat"
        cached = os.path.join(cache_dir, name)
        if not offline and (force_refresh or not os.path.exists(cached)):
            if cache_server_url:
                try:
                    url = f"{cache_server_url}/{name}"
                    _download_file(url, cached)
                except Exception:
                    _download_file(source, cached)
            else:
                _download_file(source, cached)
            if encryption_key is not None:
                with open(cached, "rb") as f:
                    data = f.read()
                with open(cached, "wb") as f:
                    f.write(b"ENC" + encrypt_bytes(data, encryption_key))
        path = cached
    else:
        path = source
    data_bytes: bytes | None = None
    if encryption_key is not None and os.path.exists(path):
        with open(path, "rb") as f:
            raw = f.read()
        if raw.startswith(b"ENC"):
            data_bytes = decrypt_bytes(raw[3:], encryption_key)
        else:
            data_bytes = raw
    ext = os.path.splitext(path)[1].lower()
    if ext == ".zip":
        for attempt in range(2):
            try:
                zf_obj = (
                    zipfile.ZipFile(io.BytesIO(data_bytes))
                    if data_bytes is not None
                    else zipfile.ZipFile(path)
                )
                with zf_obj as zf:
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
                break
            except zipfile.BadZipFile:
                if (
                    attempt == 0
                    and (source.startswith("http://") or source.startswith("https://"))
                    and not offline
                ):
                    _download_file(source, path)
                    if encryption_key is not None:
                        with open(path, "rb") as f:
                            data = f.read()
                        with open(path, "wb") as f:
                            f.write(b"ENC" + encrypt_bytes(data, encryption_key))
                    if encryption_key is not None:
                        with open(path, "rb") as f:
                            raw = f.read()
                        if raw.startswith(b"ENC"):
                            data_bytes = decrypt_bytes(raw[3:], encryption_key)
                        else:
                            data_bytes = raw
                else:
                    raise
    elif ext in {".csv", ""}:
        if data_bytes is not None:
            df = pd.read_csv(io.BytesIO(data_bytes))
        else:
            df = pd.read_csv(path)
    elif ext in {".json", ".jsonl"}:
        if data_bytes is not None:
            df = pd.read_json(io.BytesIO(data_bytes), lines=ext == ".jsonl")
        else:
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
                    if encryption_key is not None:
                        with open(cached_path, "rb") as f:
                            data = f.read()
                        with open(cached_path, "wb") as f:
                            f.write(b"ENC" + encrypt_bytes(data, encryption_key))
                with open(cached_path, "rb") as f:
                    dep_bytes = f.read()
                if encryption_key is not None and dep_bytes.startswith(b"ENC"):
                    dep_bytes = decrypt_bytes(dep_bytes[3:], encryption_key)
                data_bytes = dep_bytes
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
            {
                "input": inp if dataloader is None else dataloader.decode(inp),
                "target": tgt if dataloader is None else dataloader.decode(tgt),
            },
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

    if use_cache:
        data_obj = memory_pool.allocate() if memory_pool else _DATASET_POOL.allocate()
        data_obj.extend(pairs)
        _DATASET_CACHE[cache_key] = data_obj
    if memory_manager is not None:
        est = sum(sys.getsizeof(a) + sys.getsizeof(b) for a, b in pairs)
        memory_manager.notify_allocation(est)

    if metrics_visualizer:
        metrics_visualizer.log_event("dataset_load_end", {"pairs": len(pairs)})
    global_event_bus.publish("dataset_load_end", {"pairs": len(pairs)})
    if return_deps:
        return pairs, deps
    return pairs


def load_kuzu_graph(
    db_path: str,
    query: str,
    *,
    input_column: str = "input",
    target_column: str = "target",
    limit: int | None = None,
    dataloader: "DataLoader | None" = None,
) -> list[tuple[Any, Any]]:
    """Execute ``query`` on a Kùzu graph and return ``(input, target)`` pairs.

    Parameters
    ----------
    db_path:
        Filesystem path to the Kùzu database.
    query:
        Cypher query selecting rows. The query must return columns matching
        ``input_column`` and ``target_column``.
    input_column / target_column:
        Names of the columns in the query result used as input and target
        values respectively.
    limit:
        Optional maximum number of pairs to return. ``None`` loads all rows.
    dataloader:
        Optional :class:`~marble.DataLoader` used to encode the raw values.
    """

    with KuzuGraphDatabase(db_path) as db:
        rows = db.execute(query)

    pairs: list[tuple[Any, Any]] = []
    for row in rows:
        inp = row.get(input_column)
        tgt = row.get(target_column)
        if dataloader is not None:
            inp = dataloader.encode(inp)
            tgt = dataloader.encode(tgt)
        pairs.append((inp, tgt))
        if limit is not None and len(pairs) >= limit:
            break
    return pairs


def load_training_data_from_config(
    dataset_cfg: dict,
    *,
    dataloader: "DataLoader | None" = None,
) -> list[tuple[Any, Any]]:
    """Load training pairs according to ``dataset`` configuration section.

    When ``dataset_cfg`` contains ``use_kuzu_graph: true`` the function reads
    data from a Kùzu graph using :func:`load_kuzu_graph`. Otherwise it falls back
    to :func:`load_dataset` and expects ``dataset_cfg['source']`` to reference a
    conventional dataset file or URL.
    """

    if dataset_cfg.get("use_kuzu_graph"):
        graph_cfg = dataset_cfg.get("kuzu_graph", {})
        db_path = graph_cfg.get("db_path")
        query = graph_cfg.get("query")
        if not db_path or not query:
            raise ValueError(
                "dataset.kuzu_graph.db_path and dataset.kuzu_graph.query must be provided"
            )
        return load_kuzu_graph(
            db_path,
            query,
            input_column=graph_cfg.get("input_column", "input"),
            target_column=graph_cfg.get("target_column", "target"),
            limit=graph_cfg.get("limit"),
            dataloader=dataloader,
        )

    source = dataset_cfg.get("source")
    if source is None:
        raise ValueError("dataset.source must be specified when not using a Kùzu graph")

    kwargs: dict[str, Any] = {}
    for key in [
        "cache_dir",
        "input_col",
        "target_col",
        "limit",
        "force_refresh",
        "offline",
        "num_shards",
        "shard_index",
        "cache_url",
        "encryption_key",
    ]:
        if key in dataset_cfg:
            if key == "cache_url":
                kwargs["cache_server_url"] = dataset_cfg[key]
            else:
                kwargs[key] = dataset_cfg[key]

    pairs = load_dataset(source, dataloader=dataloader, **kwargs)
    registry = dataset_cfg.get("version_registry")
    version = dataset_cfg.get("version")
    if registry and version:
        from dataset_versioning import apply_version

        pairs = apply_version(pairs, registry, version)
    return pairs


class StreamingCSVLoader:
    """Iterate over a CSV file line by line with resume support.

    The loader stores the current byte ``offset`` in ``<path>.meta.json`` after
    each yielded row so subsequent runs can resume where the previous one left
    off. When ``tokenizer`` is provided, an ``input_ids`` field containing token
    ids is added to each row using :func:`tokenize_line`.
    """

    def __init__(
        self,
        path: str,
        *,
        tokenizer: Any | None = None,
        meta_suffix: str = ".meta.json",
    ) -> None:
        self.path = path
        self.meta_path = path + meta_suffix
        self.tokenizer = tokenizer
        self._file = open(path, "r", encoding="utf-8", newline="")
        self._header = self._file.readline().strip().split(",")
        if os.path.exists(self.meta_path):
            try:
                with open(self.meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                self._file.seek(meta.get("offset", 0))
                if self._file.tell() == 0:
                    self._file.readline()
            except Exception:
                pass

    def __iter__(self):
        while True:
            line = self._file.readline()
            if not line:
                break
            offset = self._file.tell()
            with open(self.meta_path, "w", encoding="utf-8") as f:
                json.dump({"offset": offset}, f)
            row = next(csv.DictReader([line], fieldnames=self._header))
            if self.tokenizer and "input" in row:
                row["input_ids"] = tokenize_line(self.tokenizer, row["input"])
            yield row

    def close(self) -> None:
        self._file.close()


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


def clear_dataset_cache() -> None:
    """Release cached datasets back to the memory pool."""
    for data in list(_DATASET_CACHE.values()):
        data.clear()
        _DATASET_POOL.release(data)
    _DATASET_CACHE.clear()
