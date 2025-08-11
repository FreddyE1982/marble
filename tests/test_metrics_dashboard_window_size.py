import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from marble_main import MARBLE
from metrics_dashboard import MetricsDashboard
from tests.test_core_functions import minimal_params


def test_metrics_dashboard_window_size_config(monkeypatch):
    captured = {}

    orig_init = MetricsDashboard.__init__

    def fake_init(self, mv, host="localhost", port=8050, update_interval=1000, window_size=10):
        captured["window_size"] = window_size
        orig_init(self, mv, host=host, port=port, update_interval=update_interval, window_size=window_size)

    monkeypatch.setattr(MetricsDashboard, "__init__", fake_init)
    monkeypatch.setattr(MetricsDashboard, "start", lambda self: None)

    params = minimal_params()
    MARBLE(params, dashboard_params={"enabled": True, "window_size": 42})

    assert captured["window_size"] == 42
