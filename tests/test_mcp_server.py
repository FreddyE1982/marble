import asyncio
import aiohttp
import time

from marble_brain import Brain
from marble_core import Core, DataLoader
from marble_neuronenblitz import Neuronenblitz
from prompt_memory import PromptMemory
from tests.test_core_functions import minimal_params

from mcp_server import MCPServer


def test_mcp_server_infer():
    params = minimal_params()
    core = Core(params)
    nb = Neuronenblitz(core)
    brain = Brain(core, nb, DataLoader())
    server = MCPServer(brain, host="localhost", port=5080)
    server.start()
    time.sleep(0.5)
    try:
        async def _request():
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "http://localhost:5080/mcp/infer", json={"input": 0.1}
                ) as resp:
                    assert resp.status == 200
                    data = await resp.json()
                    assert "output" in data
                    assert isinstance(data["output"], float)
        asyncio.run(_request())
    finally:
        server.stop()


def test_mcp_server_context(tmp_path):
    params = minimal_params()
    core = Core(params)
    nb = Neuronenblitz(core)
    brain = Brain(core, nb, DataLoader())
    memory = PromptMemory(max_size=2)
    server = MCPServer(brain, host="localhost", port=5081, prompt_memory=memory)
    server.start()
    time.sleep(0.5)
    try:
        async def _request():
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "http://localhost:5081/mcp/context", json={"text": "hello"}
                ) as resp:
                    assert resp.status == 200
                    data = await resp.json()
                    assert "output" in data
        asyncio.run(_request())
        assert memory.get_pairs()[0][0] == "hello"
    finally:
        server.stop()


def test_mcp_server_token_auth():
    params = minimal_params()
    core = Core(params)
    nb = Neuronenblitz(core)
    brain = Brain(core, nb, DataLoader())
    server = MCPServer(brain, host="localhost", port=5082, auth_token="secret")
    server.start()
    time.sleep(0.5)
    try:
        async def _request(headers=None):
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "http://localhost:5082/mcp/infer",
                    json={"input": 0.2},
                    headers=headers,
                ) as resp:
                    return resp.status

        # Missing token should fail
        assert asyncio.run(_request()) == 401
        # Correct token succeeds
        assert (
            asyncio.run(_request({"Authorization": "Bearer secret"})) == 200
        )
    finally:
        server.stop()


def test_mcp_server_basic_auth():
    params = minimal_params()
    core = Core(params)
    nb = Neuronenblitz(core)
    brain = Brain(core, nb, DataLoader())
    server = MCPServer(
        brain,
        host="localhost",
        port=5083,
        auth_username="user",
        auth_password="pass",
    )
    server.start()
    time.sleep(0.5)
    try:
        async def _request(headers=None):
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "http://localhost:5083/mcp/infer",
                    json={"input": 0.3},
                    headers=headers,
                ) as resp:
                    return resp.status

        import base64

        creds = base64.b64encode(b"user:pass").decode()
        auth_header = {"Authorization": f"Basic {creds}"}
        assert asyncio.run(_request(auth_header)) == 200
        assert asyncio.run(_request()) == 401
    finally:
        server.stop()
