import os, sys
import http.server
import socketserver
import threading
import pandas as pd
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from dataset_loader import load_dataset, prefetch_dataset
from pipeline import Pipeline

class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, files=None, **kwargs):
        self.files = files or {}
        super().__init__(*args, **kwargs)

    def do_GET(self):
        if self.path in self.files:
            content = self.files[self.path]
            self.send_response(200)
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)
        else:
            self.send_response(404)
            self.end_headers()


def test_prefetch_pipeline(tmp_path):
    files = {
        "/input.txt": b"hello",
        "/target.txt": b"world",
    }
    dataset_csv = "input,target\nhttp://localhost:{port}/input.txt,http://localhost:{port}/target.txt\n"
    with socketserver.TCPServer(("localhost", 0), lambda *a, **k: Handler(*a, files=files, **k)) as httpd:
        port = httpd.server_address[1]
        dataset_content = dataset_csv.format(port=port).encode()
        files["/data.csv"] = dataset_content
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        url = f"http://localhost:{port}/data.csv"
        prefetch_dataset(url, cache_dir=tmp_path)
        pipe = Pipeline([
            {"func": "load_dataset", "module": "dataset_loader", "params": {"source": url, "cache_dir": str(tmp_path)}}
        ])
        result = pipe.execute()
        httpd.shutdown()
        thread.join()
    assert result[0] == [(b"hello", b"world")]
