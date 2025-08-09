from message_bus import MessageBus
from wanderer_messages import PathUpdate
from remote_wanderer import RemoteWandererClient, RemoteWandererServer


import torch


def dummy_explore(seed: int, max_steps: int, device: str | None = None):
    torch.manual_seed(seed)
    nodes = list(range(max_steps))
    score = float(torch.rand(1, device=device if device and torch.cuda.is_available() else None).cpu().item())
    yield PathUpdate(nodes=nodes, score=score)


def test_remote_wanderer_round_trip():
    bus = MessageBus()
    server = RemoteWandererServer(bus, "coord")
    client = RemoteWandererClient(bus, "w1", dummy_explore)
    client.start()
    result = server.request_exploration("w1", seed=42, max_steps=3)
    assert result.wanderer_id == "w1"
    assert len(result.paths) == 1
    assert result.paths[0].nodes == [0, 1, 2]
    assert result.device in {"cpu", "cuda:0"}
    client.stop()
