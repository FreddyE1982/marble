from concurrent.futures import ThreadPoolExecutor

from message_bus import MessageBus
from wanderer_messages import PathUpdate
from remote_wanderer import RemoteWandererClient, RemoteWandererServer


import torch


def dummy_explore(seed: int, max_steps: int, device: str | None = None):
    torch.manual_seed(seed)
    nodes = list(range(max_steps))
    score = float(
        torch.rand(1, device=device if device and torch.cuda.is_available() else None)
        .cpu()
        .item()
    )
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


def test_multiple_remote_wanderers_exchange():
    bus = MessageBus()
    server = RemoteWandererServer(bus, "coord")
    client1 = RemoteWandererClient(bus, "w1", dummy_explore)
    client2 = RemoteWandererClient(bus, "w2", dummy_explore)
    client1.start()
    client2.start()

    def request(wid: str, seed: int):
        return server.request_exploration(wid, seed=seed, max_steps=3)

    with ThreadPoolExecutor(max_workers=2) as ex:
        fut1 = ex.submit(request, "w1", 42)
        fut2 = ex.submit(request, "w2", 43)
        res1 = fut1.result(timeout=5)
        res2 = fut2.result(timeout=5)

    assert res1.wanderer_id == "w1"
    assert res2.wanderer_id == "w2"
    assert res1.paths[0].nodes == [0, 1, 2]
    assert res2.paths[0].nodes == [0, 1, 2]
    assert res1.device in {"cpu", "cuda:0"}
    assert res2.device in {"cpu", "cuda:0"}

    client1.stop()
    client2.stop()
