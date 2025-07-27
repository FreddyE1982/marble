from experiment_tracker import ExperimentTracker
from marble_base import MetricsVisualizer

class DummyTracker(ExperimentTracker):
    def __init__(self):
        self.logged = []

    def log_metrics(self, metrics, step):
        self.logged.append((step, metrics))


def test_metrics_visualizer_logs_to_tracker():
    tracker = DummyTracker()
    mv = MetricsVisualizer(tracker=tracker)
    mv.update({"loss": 0.5, "vram_usage": 10})
    mv.update({"loss": 0.4, "vram_usage": 11})
    assert len(tracker.logged) == 2
    assert tracker.logged[0][1]["loss"] == 0.5
    mv.close()
