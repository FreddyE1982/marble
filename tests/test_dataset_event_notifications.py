from pipeline import Pipeline
from event_bus import global_event_bus


def test_dataset_events_emitted(tmp_path):
    events = []

    def listener(name, data):
        events.append((name, data))

    global_event_bus.subscribe(
        listener, events=["dataset_load_start", "dataset_load_end"]
    )
    csv_path = tmp_path / "d.csv"
    csv_path.write_text("input,target\n1,2\n3,4\n")
    pipe = Pipeline(
        [
            {
                "module": "dataset_loader",
                "func": "load_dataset",
                "params": {"source": str(csv_path)},
            }
        ]
    )
    pipe.execute()
    names = [n for n, _ in events]
    assert "dataset_load_start" in names
    assert "dataset_load_end" in names
    end_payload = [d for n, d in events if n == "dataset_load_end"][0]
    assert end_payload["pairs"] == 2

