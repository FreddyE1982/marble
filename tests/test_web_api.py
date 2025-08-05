import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import requests

from marble_brain import Brain
from marble_core import Core, DataLoader
from marble_neuronenblitz import Neuronenblitz
from prompt_memory import PromptMemory
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
        resp = requests.post(
            "http://localhost:5090/infer", json={"input": 0.1}, timeout=5
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "output" in data
        assert isinstance(data["output"], float)
    finally:
        server.stop()


def test_inference_server_with_prompt_memory(tmp_path):
    params = minimal_params()
    core = Core(params)
    nb = Neuronenblitz(core)
    brain = Brain(core, nb, DataLoader())
    memory = PromptMemory(max_size=2)
    server = InferenceServer(brain, host="localhost", port=5091, prompt_memory=memory)
    server.start()
    try:
        time.sleep(0.5)
        resp = requests.post(
            "http://localhost:5091/infer", json={"text": "hello"}, timeout=5
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "output" in data
        assert len(memory.get_pairs()) == 1
        assert memory.get_pairs()[0][0] == "hello"
    finally:
        server.stop()


def test_graph_endpoint(tmp_path):
    params = minimal_params()
    core = Core(params)
    nb = Neuronenblitz(core)
    brain = Brain(core, nb, DataLoader())
    server = InferenceServer(brain, host="localhost", port=5092)
    server.start()
    try:
        time.sleep(0.5)
        resp = requests.get("http://localhost:5092/graph", timeout=5)
        assert resp.status_code == 200
        data = resp.json()
        assert "nodes" in data and "edges" in data
    finally:
        server.stop()


def test_graph_endpoint_auth(tmp_path):
    params = minimal_params()
    core = Core(params)
    nb = Neuronenblitz(core)
    brain = Brain(core, nb, DataLoader())
    server = InferenceServer(brain, host="localhost", port=5093, api_token="secret")
    server.start()
    try:
        time.sleep(0.5)
        resp = requests.get("http://localhost:5093/graph", timeout=5)
        assert resp.status_code == 401
        resp = requests.get(
            "http://localhost:5093/graph",
            headers={"Authorization": "Bearer secret"},
            timeout=5,
        )
        assert resp.status_code == 200
    finally:
        server.stop()
