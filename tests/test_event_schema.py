import types

from dataset_loader import load_dataset
from event_bus import PROGRESS_EVENT, convert_legacy_events, global_event_bus
from pipeline import Pipeline


def test_events_include_name_and_timestamp(tmp_path):
    dummy = types.ModuleType("dummy_mod")

    def inc(x: int) -> int:
        return x + 1

    dummy.inc = inc
    import sys

    sys.modules["dummy_mod"] = dummy

    events = []

    def listener(name, data):
        events.append(data)

    global_event_bus.subscribe(listener, events=[PROGRESS_EVENT, "dataset_load_start"])

    Pipeline([{"module": "dummy_mod", "func": "inc", "params": {"x": 1}}]).execute()
    path = tmp_path / "d.csv"
    path.write_text("input,target\n1,2\n")
    load_dataset(str(path))

    assert all("name" in e and "timestamp" in e for e in events)


def test_convert_legacy_events_adds_schema():
    legacy = [("old", {"value": 1})]
    new = convert_legacy_events(legacy)
    assert new[0]["name"] == "old"
    assert "timestamp" in new[0]
