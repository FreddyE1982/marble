import os
import sys
import threading
from http.server import HTTPServer, SimpleHTTPRequestHandler
from functools import partial

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dataset_loader import load_dataset


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
