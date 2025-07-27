from marble_utils import core_to_json, core_from_json
from marble_neuronenblitz import Neuronenblitz
from marble_brain import Brain
from marble_core import DataLoader
from data_compressor import DataCompressor
import base64
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
import time
import requests
from collections import deque

class RemoteBrainServer:
    """HTTP server hosting a full remote MARBLE brain."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 8000,
        remote_url: str | None = None,
        compression_level: int = 6,
        compression_enabled: bool = True,
    ) -> None:
        self.host = host
        self.port = port
        self.remote_client = (
            RemoteBrainClient(remote_url, compression_level=compression_level,
                               compression_enabled=compression_enabled)
            if remote_url
            else None
        )
        self.compressor = DataCompressor(
            level=compression_level, compression_enabled=compression_enabled
        )
        self.use_compression = compression_enabled
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
                    if server.use_compression:
                        comp_b64 = payload['core']
                        comp_bytes = base64.b64decode(comp_b64.encode())
                        json_bytes = server.compressor.decompress(comp_bytes)
                        core_json = json_bytes.decode()
                    else:
                        core_json = json.dumps(payload['core'])
                    server.core = core_from_json(core_json)
                    server.neuronenblitz = Neuronenblitz(
                        server.core, remote_client=server.remote_client
                    )
                    server.brain = Brain(
                        server.core,
                        server.neuronenblitz,
                        DataLoader(),
                        remote_client=server.remote_client,
                    )
                    self._set_headers()
                    self.wfile.write(b'{}')
                elif self.path == '/process':
                    if server.neuronenblitz is None:
                        self.send_response(400)
                        self.end_headers()
                        return
                    if server.use_compression:
                        comp_b64 = payload.get('value', '')
                        comp_bytes = base64.b64decode(comp_b64.encode())
                        val_bytes = server.compressor.decompress(comp_bytes)
                        value = float(json.loads(val_bytes.decode()))
                    else:
                        value = float(payload.get('value', 0.0))
                    output, _ = server.neuronenblitz.dynamic_wander(value)
                    self._set_headers()
                    if server.use_compression:
                        out_bytes = json.dumps(output).encode()
                        comp_out = server.compressor.compress(out_bytes)
                        out_b64 = base64.b64encode(comp_out).decode()
                        self.wfile.write(json.dumps({'output': out_b64}).encode())
                    else:
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

    def __init__(
        self,
        url: str,
        timeout: float = 5.0,
        max_retries: int = 3,
        compression_level: int = 6,
        compression_enabled: bool = True,
        backoff_factor: float = 0.5,
        track_latency: bool = True,
    ) -> None:
        self.url = url.rstrip('/')
        self.timeout = timeout
        self.max_retries = max_retries
        self.compressor = DataCompressor(
            level=compression_level, compression_enabled=compression_enabled
        )
        self.use_compression = compression_enabled
        self.backoff_factor = backoff_factor
        self.track_latency = track_latency
        self.latencies: deque[float] = deque(maxlen=100)

    def _post(self, path: str, payload: dict, timeout: float) -> requests.Response:
        """POST ``payload`` to ``path`` with retries."""
        for attempt in range(self.max_retries):
            try:
                start = time.monotonic()
                resp = requests.post(self.url + path, json=payload, timeout=timeout)
                latency = time.monotonic() - start
                if self.track_latency:
                    self.latencies.append(latency)
                return resp
            except requests.RequestException:
                if attempt == self.max_retries - 1:
                    raise
                time.sleep(self.backoff_factor * (2**attempt))

    def offload(self, core) -> None:
        if self.use_compression:
            core_json = core_to_json(core).encode()
            comp = self.compressor.compress(core_json)
            payload = {"core": base64.b64encode(comp).decode()}
        else:
            payload = {"core": json.loads(core_to_json(core))}
        self._post("/offload", payload, self.timeout)

    def process(self, value: float, timeout: float | None = None) -> float:
        if self.use_compression:
            val_bytes = json.dumps(value).encode()
            comp = self.compressor.compress(val_bytes)
            payload = {'value': base64.b64encode(comp).decode()}
        else:
            payload = {'value': value}
        req_timeout = timeout if timeout is not None else self.timeout
        resp = self._post("/process", payload, req_timeout)
        data = resp.json()
        if self.use_compression:
            comp_out = base64.b64decode(data['output'].encode())
            out_bytes = self.compressor.decompress(comp_out)
            return float(json.loads(out_bytes.decode()))
        return data['output']

    @property
    def average_latency(self) -> float:
        """Return the average latency of recent requests in seconds."""
        if not self.latencies:
            return 0.0
        return float(sum(self.latencies) / len(self.latencies))

    def latency_history(self) -> list[float]:
        """Return a list of recorded latencies."""
        return list(self.latencies)
