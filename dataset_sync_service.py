import os
from typing import List, Dict

from tqdm import tqdm

from dataset_loader import load_dataset, export_dataset, clear_dataset_cache
from dataset_version_cli import _to_python_pairs
from dataset_versioning import compute_diff, apply_diff


def detect_dataset_changes(local_path: str, remote_path: str) -> List[Dict]:
    """Return operations required to transform ``remote_path`` into ``local_path``.

    Both datasets are loaded via :func:`dataset_loader.load_dataset`. Missing
    files are treated as empty datasets.
    """
    base = (
        _to_python_pairs(load_dataset(remote_path)) if os.path.exists(remote_path) else []
    )
    new = _to_python_pairs(load_dataset(local_path))
    return compute_diff(base, new)


def sync_remote_dataset(local_path: str, remote_path: str) -> int:
    """Synchronize ``remote_path`` with ``local_path`` using incremental updates.

    The function computes a delta patch between the datasets and applies it to
    the remote copy. Progress is reported via :mod:`tqdm`.  The number of applied
    operations is returned.
    """
    base = (
        _to_python_pairs(load_dataset(remote_path)) if os.path.exists(remote_path) else []
    )
    new = _to_python_pairs(load_dataset(local_path))
    ops = compute_diff(base, new)
    if not ops:
        return 0
    for _ in tqdm(ops, total=len(ops), desc="sync", unit="op"):
        pass
    updated = apply_diff(base, ops)
    export_dataset(updated, remote_path)
    clear_dataset_cache()
    return len(ops)
