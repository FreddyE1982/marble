import os, sys, time
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
    brain = Brain(core, nb, DataLoader(), remote_client=client)

    # offload all neurons
    brain.lobe_manager.genesis(range(len(core.neurons)))
    brain.offload_high_attention(threshold=-1.0)

    out, path = nb.dynamic_wander(0.5)
    assert isinstance(out, float)
    server.stop()
