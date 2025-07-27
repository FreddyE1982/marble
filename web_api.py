from __future__ import annotations

import threading
from typing import Any

from flask import Flask, request, jsonify


class InferenceServer:
    """Expose a MARBLE brain for remote inference via HTTP."""

    def __init__(self, brain, host: str = "localhost", port: int = 5000) -> None:
        self.brain = brain
        self.host = host
        self.port = port
        self.app = Flask(__name__)
        self.thread: threading.Thread | None = None
        self._setup_routes()

    def _setup_routes(self) -> None:
        @self.app.post("/infer")
        def infer():
            data: dict[str, Any] = request.get_json(force=True) or {}
            value = float(data.get("input", 0.0))
            output, _ = self.brain.neuronenblitz.dynamic_wander(value)
            return jsonify({"output": output})

        @self.app.post("/shutdown")
        def shutdown():
            request.environ.get("werkzeug.server.shutdown", lambda: None)()
            return "OK"

    def start(self) -> None:
        if self.thread is None:
            self.thread = threading.Thread(
                target=self.app.run,
                kwargs={"host": self.host, "port": self.port, "debug": False},
                daemon=True,
            )
            self.thread.start()

    def stop(self) -> None:
        if self.thread is not None:
            try:
                import requests

                requests.post(f"http://{self.host}:{self.port}/shutdown", timeout=1)
            except Exception:
                pass
            self.thread.join(timeout=5)
            self.thread = None
