import asyncio
import time
import aiohttp
import torch

from marble_brain import Brain
from marble_core import Core, DataLoader
from marble_neuronenblitz import Neuronenblitz
from pipeline import Pipeline
from pipeline_plugins import PLUGIN_REGISTRY
from tests.test_core_functions import minimal_params


def _make_marble():
    params = minimal_params()
    core = Core(params)
    nb = Neuronenblitz(core)
    brain = Brain(core, nb, DataLoader())

    class MarbleStub:
        def __init__(self, b):
            self.brain = b

        def get_brain(self):
            return self.brain

    return MarbleStub(brain)


def test_mcp_serve_model_plugin_cpu():
    assert "serve_model_mcp" in PLUGIN_REGISTRY
    marble = _make_marble()
    pipe = Pipeline(
        [{"plugin": "serve_model_mcp", "params": {"host": "localhost", "port": 5082}}]
    )
    info = pipe.execute(marble)[0]
    time.sleep(0.5)
    try:
        async def _request():
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "http://localhost:5082/mcp/infer", json={"input": 0.2}
                ) as resp:
                    assert resp.status == 200
                    data = await resp.json()
                    assert "output" in data
        asyncio.run(_request())
    finally:
        info["server"].stop()


def test_mcp_serve_model_plugin_gpu():
    if not torch.cuda.is_available():
        return
    marble = _make_marble()
    pipe = Pipeline(
        [{"plugin": "serve_model_mcp", "params": {"host": "localhost", "port": 5083}}]
    )
    info = pipe.execute(marble)[0]
    time.sleep(0.5)
    try:
        async def _request():
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "http://localhost:5083/mcp/infer", json={"input": 0.3}
                ) as resp:
                    assert resp.status == 200
        asyncio.run(_request())
    finally:
        info["server"].stop()
