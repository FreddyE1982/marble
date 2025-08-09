import time
from concurrent.futures import ThreadPoolExecutor

import torch

from message_bus import MessageBus
from remote_wanderer import RemoteWandererClient, RemoteWandererServer
from wanderer_messages import PathUpdate


def dummy_explore(seed: int, max_steps: int, device: str | None = None):
    """Deterministic exploration yielding a single path.

    Uses the provided seed to generate a reproducible score. The device
    argument is optional to maintain backward compatibility with exploration
    callbacks lacking device awareness.
    """
    torch.manual_seed(seed)
    rand_kwargs = {"device": device} if device and torch.cuda.is_available() else {}
    score = float(torch.rand(1, **rand_kwargs).cpu().item())
    nodes = list(range(max_steps))
    yield PathUpdate(nodes=nodes, score=score)


def test_remote_wanderer_recovers_after_disconnect():
    """Ensure coordination succeeds if the wanderer reconnects later.

    The coordinator sends a request while the wanderer client is offline. Once
    the client starts and processes the pending request the result must still
    be delivered correctly.
    """
    bus = MessageBus()
    server = RemoteWandererServer(bus, "coord", timeout=10.0)
    bus.register("w1")  # allow queuing messages while client is offline
    client = RemoteWandererClient(bus, "w1", dummy_explore)

    def request_result():
        return server.request_exploration("w1", seed=123, max_steps=4, timeout=10.0)

    with ThreadPoolExecutor(max_workers=1) as ex:
        future = ex.submit(request_result)
        # give the server time to queue the request while client is offline
        time.sleep(0.2)
        client.start()  # simulate the wanderer reconnecting
        result = future.result(timeout=10.0)

    assert result.wanderer_id == "w1"
    assert result.paths[0].nodes == [0, 1, 2, 3]
    assert result.device in {"cpu", "cuda:0"}
    client.stop()
