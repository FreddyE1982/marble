import os, sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from marble_core import Core
from marble_base import MetricsVisualizer
from tests.test_core_functions import minimal_params


class DummyMV:
    def __init__(self):
        self.updates = []

    def update(self, metrics):
        self.updates.append(metrics)


def test_get_memory_usage_metrics():
    params = minimal_params()
    core = Core(params)
    metrics = core.get_memory_usage_metrics()
    assert metrics["vram_usage"] >= 0
    assert metrics["ram_usage"] >= 0
    assert metrics["system_memory"] > 0


def test_check_memory_usage_updates_visualizer():
    params = minimal_params()
    mv = DummyMV()
    core = Core(params, metrics_visualizer=mv)
    core.check_memory_usage()
    assert mv.updates
