import torch
from experiment_tracker import KuzuExperimentTracker, attach_tracker_to_events
from marble_base import MetricsVisualizer
from pipeline import Pipeline
from event_bus import PROGRESS_EVENT


def sample_step(device: str) -> str:
    t = torch.tensor([1.0], device=device)
    return t.device.type


def test_kuzu_tracker_gpu_compatible(tmp_path):
    db = tmp_path / "exp.kuzu"
    tracker = KuzuExperimentTracker(str(db))
    mv = MetricsVisualizer(tracker=tracker)
    mv.update({"loss": 0.5})
    detach = attach_tracker_to_events(tracker, events=[PROGRESS_EVENT])
    pipe = Pipeline()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    pipe.add_step(func="sample_step", module=__name__, params={"device": dev})
    result = pipe.execute()[0]
    detach()
    rows = tracker.db.execute("MATCH (m:Metric) RETURN count(m) AS cnt")
    assert rows[0]["cnt"] == 1
    rows = tracker.db.execute("MATCH (e:Event) RETURN count(e) AS cnt")
    assert rows[0]["cnt"] >= 1
    assert result == dev
