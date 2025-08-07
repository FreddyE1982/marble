from __future__ import annotations

"""Bridge exposing MARBLE tools via an MCP compatible HTTP endpoint."""

import asyncio
import threading
import uuid
from typing import Any
from queue import Empty

from aiohttp import web

from event_bus import global_event_bus
from message_bus import MessageBus


class MCPToolBridge:
    """Asynchronous server translating MCP tool requests to the MessageBus.

    Requests received on ``/mcp/tool`` are forwarded to the configured
    ``MessageBus`` ``target``. The bridge waits for a corresponding reply
    containing the same ``request_id`` and returns the tool result to the HTTP
    client. This allows external MCP clients to access MARBLE tools while the
    tool execution remains decoupled inside the ``ToolManagerPlugin``.
    """

    def __init__(
        self,
        bus: MessageBus,
        *,
        target: str = "tool_manager",
        host: str = "localhost",
        port: int = 8766,
        agent_id: str = "mcp_tool_bridge",
    ) -> None:
        self.bus = bus
        self.target = target
        self.host = host
        self.port = port
        self.agent_id = agent_id
        self.bus.register(self.agent_id)
        self.app = web.Application()
        self.app.router.add_post("/mcp/tool", self._handle_tool)
        self.runner: web.AppRunner | None = None
        self.loop: asyncio.AbstractEventLoop | None = None
        self.thread: threading.Thread | None = None

    async def _handle_tool(self, request: web.Request) -> web.Response:
        data: dict[str, Any] = await request.json() if request.can_read_body else {}
        query = str(data.get("query", ""))
        request_id = data.get("id", str(uuid.uuid4()))
        self.bus.send(
            self.agent_id,
            self.target,
            {"request_id": request_id, "query": query},
        )
        try:
            while True:
                msg = self.bus.receive(self.agent_id, timeout=5)
                if msg.content.get("request_id") == request_id:
                    break
        except Empty:
            return web.json_response({"error": "timeout"}, status=504)
        global_event_bus.publish("mcp_tool_request", {"query": query})
        return web.json_response(
            {"tool": msg.content.get("tool"), "result": msg.content.get("result")}
        )

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
                "mcp_tool_bridge_start", {"host": self.host, "port": self.port}
            )

    async def _stop(self) -> None:
        if self.runner is not None:
            await self.runner.cleanup()
            self.runner = None

    def stop(self) -> None:
        if self.thread and self.loop:
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
                "mcp_tool_bridge_stop", {"host": self.host, "port": self.port}
            )
