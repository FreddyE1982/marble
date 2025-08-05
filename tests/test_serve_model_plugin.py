import time

import requests
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


def test_serve_model_plugin_cpu():
    assert "serve_model" in PLUGIN_REGISTRY
    marble = _make_marble()
    pipe = Pipeline(
        [{"plugin": "serve_model", "params": {"host": "localhost", "port": 5093}}]
    )
    info = pipe.execute(marble)[0]
    time.sleep(0.5)
    try:
        resp = requests.post(
            "http://localhost:5093/infer", json={"input": 0.2}, timeout=5
        )
        assert resp.status_code == 200
        assert "output" in resp.json()
    finally:
        info["server"].stop()


def test_serve_model_plugin_gpu():
    if not torch.cuda.is_available():
        return
    marble = _make_marble()
    pipe = Pipeline(
        [{"plugin": "serve_model", "params": {"host": "localhost", "port": 5094}}]
    )
    info = pipe.execute(marble)[0]
    time.sleep(0.5)
    try:
        resp = requests.post(
            "http://localhost:5094/infer", json={"input": 0.3}, timeout=5
        )
        assert resp.status_code == 200
    finally:
        info["server"].stop()
