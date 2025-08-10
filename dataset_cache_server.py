from __future__ import annotations

import os
from threading import Thread

from flask import Flask, Response, abort, request, send_from_directory

from dataset_encryption import decrypt_bytes, load_key_from_env


class DatasetCacheServer:
    """Simple HTTP server to share cached dataset files.

    If an AES-256-GCM encryption key is provided, cached files prefixed with
    ``b"ENC"`` are transparently decrypted before serving. The key can be
    supplied directly or via the ``DATASET_ENCRYPTION_KEY`` environment
    variable.
    """

    def __init__(
        self,
        cache_dir: str = "dataset_cache",
        encryption_key: bytes | str | None = None,
    ) -> None:
        self.cache_dir = cache_dir
        self.app = Flask(__name__)
        self.app.add_url_rule("/<path:filename>", "get_file", self._get_file)
        self.app.add_url_rule("/shutdown", "shutdown", self._shutdown)
        self.thread: Thread | None = None
        self.host = "0.0.0.0"
        self.port = 5000
        if encryption_key is None:
            try:
                self.encryption_key = load_key_from_env()
            except KeyError:
                self.encryption_key = None
        else:
            self.encryption_key = encryption_key

    def _get_file(self, filename: str):
        path = os.path.join(self.cache_dir, filename)
        if not os.path.exists(path):
            abort(404)

        if self.encryption_key is None:
            return send_from_directory(self.cache_dir, filename, as_attachment=False)

        with open(path, "rb") as f:
            raw = f.read()
        if raw.startswith(b"ENC"):
            try:
                data = decrypt_bytes(raw[3:], self.encryption_key)
            except Exception:
                abort(403)
        else:
            data = raw
        return Response(data, mimetype="application/octet-stream")

    def _shutdown(self):
        func = request.environ.get("werkzeug.server.shutdown")
        if func is not None:
            func()
        return ""

    def start(self, host: str = "0.0.0.0", port: int = 5000) -> None:
        """Start the server in a background thread."""
        self.host = host
        self.port = port

        def _run() -> None:
            self.app.run(host=self.host, port=self.port, debug=False)

        self.thread = Thread(target=_run, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        """Stop the server if running."""
        if self.thread and self.thread.is_alive():
            import requests

            try:
                requests.get(f"http://{self.host}:{self.port}/shutdown")
            except Exception:
                pass
            self.thread.join(timeout=1)
            self.thread = None
