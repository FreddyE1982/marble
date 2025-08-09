import time

from message_bus import MessageBus
from remote_wanderer import RemoteWandererClient, RemoteWandererServer
from wanderer_messages import PathUpdate


def dummy_explore(seed: int, max_steps: int, device: str | None = None):
    """Simple exploration callback returning a single path.

    Parameters
    ----------
    seed:
        Random seed controlling the score.
    max_steps:
        Number of steps in the generated path.
    device:
        Execution device. Included for API compatibility; no device-specific
        operations are performed so the function works on CPU and GPU.
    """
    nodes = list(range(max_steps))
    score = float(seed)
    yield PathUpdate(nodes=nodes, score=score)


def benchmark_latency(delays=(0.0, 0.05, 0.1), runs: int = 5, max_steps: int = 3):
    """Benchmark exploration completion time under varying network latency.

    Parameters
    ----------
    delays:
        Iterable of artificial latency values in seconds.
    runs:
        Number of exploration requests to issue per latency setting.
    max_steps:
        Step count passed to the ``dummy_explore`` callback.
    """
    results: dict[float, float] = {}
    for delay in delays:
        bus = MessageBus()
        wanderer_id = "w1"
        client = RemoteWandererClient(
            bus, wanderer_id, dummy_explore, network_latency=delay, poll_interval=0.01
        )
        server = RemoteWandererServer(bus, "coord", timeout=5.0, network_latency=delay)
        client.start()
        durations = []
        for seed in range(runs):
            start = time.perf_counter()
            server.request_exploration(wanderer_id, seed=seed, max_steps=max_steps)
            durations.append(time.perf_counter() - start)
        client.stop()
        results[delay] = sum(durations) / len(durations)
    baseline = results[min(results.keys())]
    for delay, duration in results.items():
        slowdown = ((duration - baseline) / baseline) * 100 if baseline else 0.0
        print(
            f"Delay {delay:.3f}s -> {duration:.4f}s per request (+{slowdown:.1f}% vs baseline)"
        )
    return results


if __name__ == "__main__":
    benchmark_latency()
