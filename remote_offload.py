from marble_utils import core_to_json, core_from_json
from marble_neuronenblitz import Neuronenblitz
from marble_brain import Brain
from marble_core import DataLoader
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
import requests

class RemoteBrainServer:
    """HTTP server hosting a full remote MARBLE brain."""
    def __init__(self, host="localhost", port=8000, remote_url=None):
        self.host = host
        self.port = port
        self.remote_client = RemoteBrainClient(remote_url) if remote_url else None
        self.core = None
        self.neuronenblitz = None
        self.brain = None
        self.httpd = None
        self.thread = None
        
    def start(self):
        server = self

        class Handler(BaseHTTPRequestHandler):
            def _set_headers(self):
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()

            def do_POST(self):
                length = int(self.headers.get('Content-Length', 0))
                data = self.rfile.read(length).decode()
                payload = json.loads(data or '{}')
                if self.path == '/offload':
                    server.core = core_from_json(json.dumps(payload['core']))
                    server.neuronenblitz = Neuronenblitz(server.core, remote_client=server.remote_client)
                    server.brain = Brain(server.core, server.neuronenblitz, DataLoader(), remote_client=server.remote_client)
                    self._set_headers()
                    self.wfile.write(b'{}')
                elif self.path == '/process':
                    if server.neuronenblitz is None:
                        self.send_response(400)
                        self.end_headers()
                        return
                    value = float(payload.get('value', 0.0))
                    output, _ = server.neuronenblitz.dynamic_wander(value)
                    self._set_headers()
                    self.wfile.write(json.dumps({'output': output}).encode())
                else:
                    self.send_response(404)
                    self.end_headers()

        self.httpd = HTTPServer((self.host, self.port), Handler)
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    def stop(self):
        if self.httpd:
            self.httpd.shutdown()
        if self.thread:
            self.thread.join()

class RemoteBrainClient:
    """Client used by the main brain to interact with a remote brain."""

    def __init__(self, url, timeout: float = 5.0, max_retries: int = 3):
        self.url = url.rstrip('/')
        self.timeout = timeout
        self.max_retries = max_retries

    def offload(self, core):
        payload = {'core': json.loads(core_to_json(core))}
        requests.post(self.url + '/offload', json=payload, timeout=self.timeout)

    def process(self, value):
        resp = requests.post(self.url + '/process', json={'value': value}, timeout=self.timeout)
        data = resp.json()
        return data['output']
