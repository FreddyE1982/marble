import os, sys, time
from threading import Thread
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import requests
from marble_core import Core
from marble_neuronenblitz import Neuronenblitz
from marble_brain import Brain
from marble_core import DataLoader
from tests.test_core_functions import minimal_params
from web_api import InferenceServer


def test_inference_server(tmp_path):
    params = minimal_params()
    core = Core(params)
    nb = Neuronenblitz(core)
    brain = Brain(core, nb, DataLoader())
    server = InferenceServer(brain, host="localhost", port=5090)
    server.start()
    try:
        # wait briefly for server
        time.sleep(0.5)
        resp = requests.post("http://localhost:5090/infer", json={"input": 0.1}, timeout=5)
        assert resp.status_code == 200
        data = resp.json()
        assert "output" in data
        assert isinstance(data["output"], float)
    finally:
        server.stop()
