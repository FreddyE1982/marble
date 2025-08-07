from __future__ import annotations

import asyncio
import threading
from typing import Any

import torch
from aiohttp import web

from event_bus import global_event_bus
from prompt_memory import PromptMemory


class MCPServer:
    """Asynchronous MCP server exposing MARBLE inference routes.

    The server translates MCP ``inference`` and ``context`` messages into
    calls on ``brain.neuronenblitz.dynamic_wander``. Incoming requests are
    executed in background threads to avoid blocking the event loop and will
    utilise the GPU when available.
    """

    def __init__(
        self,
        brain,
        host: str = "localhost",
        port: int = 8765,
        prompt_memory: PromptMemory | None = None,
        prompt_path: str | None = None,
    ) -> None:
        self.brain = brain
        self.host = host
        self.port = port
        self.prompt_path = prompt_path
        if prompt_memory is not None:
            self.prompt_memory = prompt_memory
        elif prompt_path is not None:
            self.prompt_memory = PromptMemory.load(prompt_path)
        else:
            self.prompt_memory = None

        self.app = web.Application()
        self.app.router.add_post("/mcp/infer", self._handle_infer)
        self.app.router.add_post("/mcp/context", self._handle_context)
        self.runner: web.AppRunner | None = None
        self.loop: asyncio.AbstractEventLoop | None = None
        self.thread: threading.Thread | None = None

    # ------------------------------------------------------------------
    # internal helpers
    def _run_wander(self, value: float) -> tuple[float, str]:
        """Run dynamic_wander on the appropriate device."""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tensor = torch.tensor([value], device=device)
        out, _ = self.brain.neuronenblitz.dynamic_wander(float(tensor.item()))
        return out, device

    async def _handle_infer(self, request: web.Request) -> web.Response:
        data: dict[str, Any] = await request.json() if request.can_read_body else {}
        if "text" in data:
            text = str(data.get("text", ""))
            composite = (
                self.prompt_memory.composite_with(text)
                if self.prompt_memory is not None
                else text
            )
            value = sum(ord(c) for c in composite) / max(len(composite), 1)
            output, device = await asyncio.to_thread(self._run_wander, value)
            if self.prompt_memory is not None:
                self.prompt_memory.add(text, str(output))
        else:
            value = float(data.get("input", 0.0))
            output, device = await asyncio.to_thread(self._run_wander, value)
        global_event_bus.publish(
            "mcp_request", {"route": "infer", "device": device, "value": value}
        )
        return web.json_response({"output": output})

    async def _handle_context(self, request: web.Request) -> web.Response:
        data: dict[str, Any] = await request.json() if request.can_read_body else {}
        text = str(data.get("text", ""))
        composite = (
            self.prompt_memory.composite_with(text)
            if self.prompt_memory is not None
            else text
        )
        value = sum(ord(c) for c in composite) / max(len(composite), 1)
        output, device = await asyncio.to_thread(self._run_wander, value)
        if self.prompt_memory is not None:
            self.prompt_memory.add(text, str(output))
        global_event_bus.publish(
            "mcp_request", {"route": "context", "device": device, "value": value}
        )
        return web.json_response({"output": output})

    # ------------------------------------------------------------------
    async def _start(self) -> None:
        self.runner = web.AppRunner(self.app)
        await self.runner.setup()
        site = web.TCPSite(self.runner, self.host, self.port)
        await site.start()

    def start(self) -> None:
        if self.thread is None:
            self.loop = asyncio.new_event_loop()

            def runner() -> None:
                asyncio.set_event_loop(self.loop)
                self.loop.run_until_complete(self._start())
                self.loop.run_forever()

            self.thread = threading.Thread(target=runner, daemon=True)
            self.thread.start()
            global_event_bus.publish(
                "mcp_server_start", {"host": self.host, "port": self.port}
            )

    async def _stop(self) -> None:
        if self.runner is not None:
            await self.runner.cleanup()
            self.runner = None

    def stop(self) -> None:
        if self.thread and self.loop:
            if self.prompt_memory is not None and self.prompt_path is not None:
                try:
                    self.prompt_memory.serialize(self.prompt_path)
                except Exception:
                    pass
            fut = asyncio.run_coroutine_threadsafe(self._stop(), self.loop)
            try:
                fut.result(5)
            except Exception:
                pass
            self.loop.call_soon_threadsafe(self.loop.stop)
            self.thread.join(timeout=5)
            self.thread = None
            self.loop = None
            global_event_bus.publish(
                "mcp_server_stop", {"host": self.host, "port": self.port}
            )
