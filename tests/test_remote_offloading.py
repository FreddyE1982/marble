import os, sys, time
import requests
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from marble_core import Core, DataLoader
from marble_neuronenblitz import Neuronenblitz
from marble_brain import Brain
from remote_offload import RemoteBrainServer, RemoteBrainClient
from tests.test_core_functions import minimal_params


def test_remote_offload_roundtrip():
    server = RemoteBrainServer(port=8001)
    server.start()
    client = RemoteBrainClient("http://localhost:8001")

    params = minimal_params()
    core = Core(params)
    nb = Neuronenblitz(core, remote_client=client)
    brain = Brain(core, nb, DataLoader(), remote_client=client, offload_enabled=True)

    # offload all neurons
    brain.lobe_manager.genesis(range(len(core.neurons)))
    brain.offload_high_attention(threshold=-1.0)

    out, path = nb.dynamic_wander(0.5)
    assert isinstance(out, float)
    assert isinstance(server.brain, Brain)
    server.stop()


def test_remote_brain_offload_chain():
    server1 = RemoteBrainServer(port=8002)
    server1.start()
    client1 = RemoteBrainClient("http://localhost:8002")

    server2 = RemoteBrainServer(port=8003)
    server2.start()
    client2 = RemoteBrainClient("http://localhost:8003")

    params = minimal_params()
    core = Core(params)
    nb = Neuronenblitz(core, remote_client=client1)
    brain = Brain(core, nb, DataLoader(), remote_client=client1, offload_enabled=True)

    brain.lobe_manager.genesis(range(len(core.neurons)))
    brain.offload_high_attention(threshold=-1.0)

    server1.brain.remote_client = client2
    server1.brain.offload_enabled = True
    server1.brain.lobe_manager.genesis(range(len(server1.core.neurons)))
    server1.brain.offload_high_attention(threshold=-1.0)

    out, _ = nb.dynamic_wander(0.2)
    assert isinstance(out, float)
    assert isinstance(server2.brain, Brain)

    server1.stop()
    server2.stop()


def test_remote_offload_uncompressed():
    server = RemoteBrainServer(port=8004, compression_enabled=False)
    server.start()
    client = RemoteBrainClient("http://localhost:8004", compression_enabled=False)

    params = minimal_params()
    core = Core(params)
    nb = Neuronenblitz(core, remote_client=client)
    brain = Brain(core, nb, DataLoader(), remote_client=client, offload_enabled=True)

    brain.lobe_manager.genesis(range(len(core.neurons)))
    brain.offload_high_attention(threshold=-1.0)

    out, _ = nb.dynamic_wander(0.3)
    assert isinstance(out, float)
    server.stop()


def test_remote_client_retries(monkeypatch):
    attempts = {"n": 0}

    def fake_post(url, json=None, timeout=0):
        attempts["n"] += 1
        if attempts["n"] < 2:
            raise requests.RequestException("fail")

        class Res:
            def json(self):
                return {"output": 1.0}

        return Res()

    monkeypatch.setattr(requests, "post", fake_post)
    client = RemoteBrainClient("http://localhost", max_retries=2, compression_enabled=False)
    val = client.process(0.2)
    assert val == 1.0
    assert attempts["n"] == 2


def test_bandwidth_and_route(monkeypatch):
    def fake_post(url, data=None, timeout=0, headers=None):
        class Res:
            headers = {"Content-Length": str(len(data))}

            def json(self):
                return {"output": 1.0}

        return Res()

    def fake_get(url, timeout=0):
        class Res:
            pass

        if "a" in url:
            time.sleep(0.02)
        else:
            time.sleep(0.01)
        return Res()

    monkeypatch.setattr("requests.post", fake_post)
    monkeypatch.setattr("requests.get", fake_get)
    client = RemoteBrainClient("http://a", compression_enabled=False)
    client.process(0.2)
    bw1 = client.average_bandwidth
    new_url = client.optimize_route(["http://a", "http://b"])  # should select b
    assert new_url == "http://b"
    client.process(0.3)
    assert client.average_bandwidth > 0
