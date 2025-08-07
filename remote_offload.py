from marble_utils import core_to_json, core_from_json
from marble_neuronenblitz import Neuronenblitz
from marble_brain import Brain
from marble_core import DataLoader
from data_compressor import DataCompressor
from crypto_utils import constant_time_compare
import base64
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
import ssl
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
        compression_algorithm: str = "zlib",
        auth_token: str | None = None,
        ssl_enabled: bool = False,
        ssl_cert_file: str | None = None,
        ssl_key_file: str | None = None,
    ) -> None:
        self.host = host
        self.port = port
        self.auth_token = auth_token
        self.remote_client = (
            RemoteBrainClient(
                remote_url,
                compression_level=compression_level,
                compression_enabled=compression_enabled,
                compression_algorithm=compression_algorithm,
            )
            if remote_url
            else None
        )
        self.compressor = DataCompressor(
            level=compression_level,
            compression_enabled=compression_enabled,
            algorithm=compression_algorithm,
        )
        self.use_compression = compression_enabled
        self.ssl_enabled = ssl_enabled
        self.ssl_cert_file = ssl_cert_file
        self.ssl_key_file = ssl_key_file
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

            def _authorized(self) -> bool:
                if server.auth_token is None:
                    return True
                header = self.headers.get('Authorization', '')
                return constant_time_compare(
                    header, f"Bearer {server.auth_token}"
                )

            def do_POST(self):
                if not self._authorized():
                    self.send_response(401)
                    self.end_headers()
                    return
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

            def do_GET(self):
                if not self._authorized():
                    self.send_response(401)
                    self.end_headers()
                    return
                if self.path == '/ping':
                    self._set_headers()
                    self.wfile.write(b'{}')
                else:
                    self.send_response(404)
                    self.end_headers()

        self.httpd = HTTPServer((self.host, self.port), Handler)
        if self.ssl_enabled:
            if not self.ssl_cert_file or not self.ssl_key_file:
                raise ValueError("ssl_cert_file and ssl_key_file required when ssl_enabled")
            context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            context.load_cert_chain(self.ssl_cert_file, self.ssl_key_file)
            self.httpd.socket = context.wrap_socket(self.httpd.socket, server_side=True)
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
        compression_algorithm: str = "zlib",
        backoff_factor: float = 0.5,
        track_latency: bool = True,
        auth_token: str | None = None,
        *,
        connect_retry_interval: float = 5.0,
        heartbeat_timeout: float = 10.0,
        ssl_verify: bool = True,
    ) -> None:
        self.url = url.rstrip('/')
        self.timeout = timeout
        self.max_retries = max_retries
        self.auth_token = auth_token
        self.compressor = DataCompressor(
            level=compression_level,
            compression_enabled=compression_enabled,
            algorithm=compression_algorithm,
        )
        self.use_compression = compression_enabled
        self.backoff_factor = backoff_factor
        self.track_latency = track_latency
        self.latencies: deque[float] = deque(maxlen=100)
        self.bytes_sent = 0
        self.bytes_received = 0
        self._bandwidth_start = time.monotonic()
        self.connect_retry_interval = connect_retry_interval
        self.heartbeat_timeout = heartbeat_timeout
        self.ssl_verify = ssl_verify
        self._connected = False

    def ping(self, timeout: float | None = None) -> None:
        """Send a heartbeat request to the remote server."""
        ping_timeout = timeout if timeout is not None else self.heartbeat_timeout
        headers = {}
        if self.auth_token is not None:
            headers["Authorization"] = f"Bearer {self.auth_token}"
        resp = requests.get(
            self.url + "/ping",
            timeout=ping_timeout,
            headers=headers,
            verify=self.ssl_verify,
        )
        if resp.status_code >= 400:
            resp.raise_for_status()

    def connect(self) -> None:
        """Ensure the remote server is reachable by pinging with retries."""
        for attempt in range(self.max_retries):
            try:
                self.ping()
                self._connected = True
                return
            except requests.RequestException:
                if attempt == self.max_retries - 1:
                    raise
                time.sleep(self.connect_retry_interval)

    def _post(
        self,
        path: str,
        payload: dict,
        timeout: float,
        *,
        retries: int | None = None,
        backoff: float | None = None,
    ) -> requests.Response:
        """POST ``payload`` to ``path`` with retries.

        Parameters
        ----------
        path:
            Endpoint path (``/offload`` or ``/process``).
        payload:
            JSON-serialisable payload.
        timeout:
            Request timeout in seconds.
        retries:
            Optional override for retry count. Defaults to ``self.max_retries``.
        backoff:
            Optional override for backoff multiplier. Defaults to
            ``self.backoff_factor``.
        """

        max_retries = retries if retries is not None else self.max_retries
        backoff_factor = backoff if backoff is not None else self.backoff_factor

        for attempt in range(max_retries):
            try:
                payload_bytes = json.dumps(payload).encode()
                start = time.monotonic()
                headers = {}
                if self.auth_token is not None:
                    headers["Authorization"] = f"Bearer {self.auth_token}"
                resp = requests.post(
                    self.url + path,
                    json=payload,
                    timeout=timeout,
                    headers=headers,
                    verify=self.ssl_verify,
                )
                if resp.status_code >= 400:
                    resp.raise_for_status()
                latency = time.monotonic() - start
                self.bytes_sent += len(payload_bytes)
                headers = getattr(resp, "headers", {})
                self.bytes_received += int(headers.get("Content-Length", "0"))
                if self.track_latency:
                    self.latencies.append(latency)
                return resp
            except requests.RequestException:
                if attempt == max_retries - 1:
                    raise
                time.sleep(backoff_factor * (2**attempt))

    def offload(
        self,
        core,
        *,
        retries: int | None = None,
        backoff: float | None = None,
    ) -> None:
        if self.use_compression:
            core_json = core_to_json(core).encode()
            comp = self.compressor.compress(core_json)
            payload = {"core": base64.b64encode(comp).decode()}
        else:
            payload = {"core": json.loads(core_to_json(core))}
        self._post("/offload", payload, self.timeout, retries=retries, backoff=backoff)

    def process(
        self,
        value: float,
        timeout: float | None = None,
        *,
        retries: int | None = None,
        backoff: float | None = None,
    ) -> float:
        if self.use_compression:
            val_bytes = json.dumps(value).encode()
            comp = self.compressor.compress(val_bytes)
            payload = {'value': base64.b64encode(comp).decode()}
        else:
            payload = {'value': value}
        req_timeout = timeout if timeout is not None else self.timeout
        resp = self._post(
            "/process",
            payload,
            req_timeout,
            retries=retries,
            backoff=backoff,
        )
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

    def optimize_route(self, urls: list[str]) -> str:
        """Select the fastest URL from ``urls`` based on ping latency."""
        best_url = self.url
        best_latency = float('inf')
        for u in urls:
            try:
                start = time.monotonic()
                headers = (
                    {"Authorization": f"Bearer {self.auth_token}"}
                    if self.auth_token is not None
                    else None
                )
                requests.get(u.rstrip('/') + '/ping', timeout=1.0, headers=headers)
                lat = time.monotonic() - start
                if lat < best_latency:
                    best_latency = lat
                    best_url = u.rstrip('/')
            except requests.RequestException:
                continue
        self.url = best_url
        return best_url

    @property
    def average_bandwidth(self) -> float:
        """Return average transfer rate in bytes/second since instantiation."""
        elapsed = max(1e-6, time.monotonic() - self._bandwidth_start)
        return float((self.bytes_sent + self.bytes_received) / elapsed)
