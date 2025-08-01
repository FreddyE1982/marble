from __future__ import annotations

import os
from threading import Thread
from flask import Flask, send_from_directory, abort, request


class DatasetCacheServer:
    """Simple HTTP server to share cached dataset files."""

    def __init__(self, cache_dir: str = "dataset_cache") -> None:
        self.cache_dir = cache_dir
        self.app = Flask(__name__)
        self.app.add_url_rule("/<path:filename>", "get_file", self._get_file)
        self.app.add_url_rule("/shutdown", "shutdown", self._shutdown)
        self.thread: Thread | None = None
        self.host = "0.0.0.0"
        self.port = 5000

    def _get_file(self, filename: str):
        path = os.path.join(self.cache_dir, filename)
        if os.path.exists(path):
            return send_from_directory(self.cache_dir, filename, as_attachment=False)
        abort(404)

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
