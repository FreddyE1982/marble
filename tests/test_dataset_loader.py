import os
import sys
import threading
from http.server import HTTPServer, SimpleHTTPRequestHandler
from functools import partial
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dataset_loader import load_dataset, prefetch_dataset, export_dataset
from marble import DataLoader


def _serve_directory(directory, port):
    handler = partial(SimpleHTTPRequestHandler, directory=directory)
    httpd = HTTPServer(("localhost", port), handler)
    thread = threading.Thread(target=httpd.serve_forever)
    thread.daemon = True
    thread.start()
    return httpd, thread


def test_load_local_csv(tmp_path):
    csv_path = tmp_path / "data.csv"
    csv_path.write_text("input,target\n1,2\n3,4\n")
    pairs = load_dataset(str(csv_path))
    assert pairs == [(1, 2), (3, 4)]


def test_load_remote_csv(tmp_path):
    csv_path = tmp_path / "remote.csv"
    csv_path.write_text("input,target\n5,6\n7,8\n")
    httpd, thread = _serve_directory(tmp_path, 9000)
    try:
        url = f"http://localhost:9000/{csv_path.name}"
        pairs = load_dataset(url, cache_dir=tmp_path / "cache")
        assert pairs == [(5, 6), (7, 8)]
    finally:
        httpd.shutdown()
        thread.join()


def test_download_progress_bar(monkeypatch, tmp_path):
    csv_path = tmp_path / "file.csv"
    csv_path.write_text("input,target\n1,1\n")
    httpd, thread = _serve_directory(tmp_path, 9010)
    updates = []

    class Dummy:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            pass

        def update(self, n):
            updates.append(n)

    monkeypatch.setattr("dataset_loader.tqdm", Dummy)
    try:
        url = f"http://localhost:9010/{csv_path.name}"
        load_dataset(url, cache_dir=tmp_path / "cache", force_refresh=True)
        assert sum(updates) > 0
    finally:
        httpd.shutdown()
        thread.join()


def test_load_zipped_csv(tmp_path):
    csv_path = tmp_path / "inner.csv"
    csv_path.write_text("input,target\n9,10\n")
    zip_path = tmp_path / "archive.zip"
    import zipfile
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.write(csv_path, arcname="inner.csv")
    pairs = load_dataset(str(zip_path))
    assert pairs == [(9, 10)]


def test_load_zipped_json(tmp_path):
    json_path = tmp_path / "inner.json"
    json_path.write_text('[{"input": 1, "target": 2}]')
    zip_path = tmp_path / "archive_json.zip"
    import zipfile
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.write(json_path, arcname="inner.json")
    pairs = load_dataset(str(zip_path))
    assert pairs == [(1, 2)]


def test_dataset_sharding(tmp_path):
    csv_path = tmp_path / "shard.csv"
    csv_path.write_text("input,target\n1,2\n3,4\n5,6\n7,8\n")
    pairs = load_dataset(str(csv_path), num_shards=2, shard_index=1)
    assert pairs == [(3, 4), (7, 8)]


def test_offline_mode(tmp_path):
    csv_path = tmp_path / "offline.csv"
    csv_path.write_text("input,target\n9,10\n")
    httpd, thread = _serve_directory(tmp_path, 9050)
    try:
        url = f"http://localhost:9050/{csv_path.name}"
        cache = tmp_path / "cache"
        # first download to cache
        pairs = load_dataset(url, cache_dir=cache)
        assert pairs == [(9, 10)]
    finally:
        httpd.shutdown()
        thread.join()

    pairs = load_dataset(url, cache_dir=cache, offline=True)
    assert pairs == [(9, 10)]
    with pytest.raises(FileNotFoundError):
        load_dataset(url, cache_dir=tmp_path / "missing", offline=True)


def test_prefetch_dataset(tmp_path):
    csv_path = tmp_path / "prefetch.csv"
    csv_path.write_text("input,target\n1,2\n")
    httpd, thread = _serve_directory(tmp_path, 9060)
    try:
        url = f"http://localhost:9060/{csv_path.name}"
        t = prefetch_dataset(url, cache_dir=tmp_path / "cache")
        t.join()
        pairs = load_dataset(url, cache_dir=tmp_path / "cache", offline=True)
        assert pairs == [(1, 2)]
    finally:
        httpd.shutdown()
        thread.join()


def test_load_dataset_with_dataloader(tmp_path):
    csv_path = tmp_path / "dl.csv"
    csv_path.write_text("input,target\nhello,world\n")
    dl = DataLoader()
    pairs = load_dataset(str(csv_path), dataloader=dl)
    inp, tgt = pairs[0]
    assert dl.decode(inp) == "hello"
    assert dl.decode(tgt) == "world"


def test_load_dataset_dependencies_and_filter(tmp_path):
    csv_path = tmp_path / "deps.csv"
    csv_path.write_text("input,target\n1,2\n3,4\n")
    pairs, deps = load_dataset(str(csv_path), return_deps=True, filter_expr="input > 1")
    assert pairs == [(3, 4)]
    assert len(deps) == 1
    assert deps[0]["input_source"] == 3
    assert deps[0]["target_source"] == 4
    assert isinstance(deps[0]["id"], str) and len(deps[0]["id"]) == 64


def test_export_dataset_roundtrip(tmp_path):
    pairs = [(5, 6), (7, 8)]
    path = tmp_path / "out.csv"
    export_dataset(pairs, str(path))
    loaded = load_dataset(str(path))
    assert loaded == pairs

