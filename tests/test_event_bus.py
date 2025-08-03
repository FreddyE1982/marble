import time
from event_bus import EventBus


def test_event_bus_filter_and_rate_limit():
    bus = EventBus()
    received: list[tuple[str, dict]] = []

    bus.subscribe(lambda n, d: received.append((n, d)), events=["alpha"], rate_limit_hz=5)
    # Event with unmatched name is ignored
    bus.publish("beta", {"v": 1})
    bus.publish("alpha", {"v": 2})
    # Immediately publishing again should be rate limited
    bus.publish("alpha", {"v": 3})
    assert received == [("alpha", {"v": 2})]
    # After waiting past the rate limit another event is delivered
    time.sleep(0.25)
    bus.publish("alpha", {"v": 4})
    assert received == [("alpha", {"v": 2}), ("alpha", {"v": 4})]


def test_event_bus_overhead():
    bus = EventBus()
    start = time.perf_counter()
    for _ in range(2000):
        bus.publish("evt")
    baseline = time.perf_counter() - start

    bus.subscribe(lambda n, d: None)
    start = time.perf_counter()
    for _ in range(2000):
        bus.publish("evt")
    hooked = time.perf_counter() - start

    # Overhead with a no-op subscriber should remain small (<0.05s)
    assert hooked - baseline < 0.05
