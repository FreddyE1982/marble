import time

from message_bus import MessageBus
from remote_wanderer import RemoteWandererClient, RemoteWandererServer
from wanderer_messages import PathUpdate


def dummy_explore(seed: int, max_steps: int, device: str | None = None):
    nodes = list(range(max_steps))
    yield PathUpdate(nodes=nodes, score=float(seed))


def test_latency_delay_is_applied():
    bus = MessageBus()
    delay = 0.05
    client = RemoteWandererClient(
        bus, "w1", dummy_explore, network_latency=delay, poll_interval=0.01
    )
    server = RemoteWandererServer(bus, "coord", timeout=2.0, network_latency=delay)
    client.start()
    start = time.perf_counter()
    server.request_exploration("w1", seed=0, max_steps=2)
    duration = time.perf_counter() - start
    client.stop()
    assert duration >= delay * 2
