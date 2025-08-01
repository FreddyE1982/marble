import http.server
import socketserver
import threading
import tempfile
from dataset_replication import replicate_dataset

class Handler(http.server.BaseHTTPRequestHandler):
    def do_POST(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b'ok')


def test_replicate_dataset(tmp_path):
    file_path = tmp_path / "ds.bin"
    file_path.write_bytes(b'content')
    with socketserver.TCPServer(("localhost", 0), Handler) as httpd:
        port = httpd.server_address[1]
        t = threading.Thread(target=httpd.serve_forever, daemon=True)
        t.start()
        replicate_dataset(str(file_path), [f"http://localhost:{port}"])
        httpd.shutdown()
        t.join()
