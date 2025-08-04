import types
import sys
from event_bus import PROGRESS_EVENT, global_event_bus
from pipeline import Pipeline


def test_pipeline_emits_progress_events():
    dummy = types.ModuleType("dummy_mod")

    def square(x: int) -> int:
        return x * x

    dummy.square = square
    sys.modules["dummy_mod"] = dummy

    steps = [
        {"module": "dummy_mod", "func": "square", "params": {"x": 2}},
        {"module": "dummy_mod", "func": "square", "params": {"x": 3}},
    ]
    events = []

    def listener(name, data):
        events.append(data)

    global_event_bus.subscribe(listener, events=[PROGRESS_EVENT])
    Pipeline(steps).execute()
    assert len(events) == 4
    first, last = events[0], events[-1]
    assert first["status"] == "started"
    assert last["status"] == "completed"
    assert last["index"] == 1 and last["total"] == 2
    assert last["device"] in {"cpu", "cuda"}
