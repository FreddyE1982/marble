import time
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from marble_base import MetricsVisualizer
from metrics_dashboard import MetricsDashboard


def test_dashboard_start_stop():
    mv = MetricsVisualizer()
    dashboard = MetricsDashboard(mv, port=8060, update_interval=200)
    dashboard.start()
    # allow server thread to start
    time.sleep(0.5)
    assert dashboard.thread is not None and dashboard.thread.is_alive()
    # Stopping simply terminates when main thread exits
    dashboard.stop()


def test_dashboard_smoothing():
    mv = MetricsVisualizer()
    mv.metrics["loss"] = [1, 2, 3, 4]
    dashboard = MetricsDashboard(mv, port=8061, update_interval=200, window_size=2)
    assert dashboard.smooth([1, 2, 3])[-1] <= 2.5


def test_build_figure_select_metrics():
    mv = MetricsVisualizer()
    mv.metrics["loss"] = [1, 2, 3]
    mv.metrics["reward"] = [0.1, 0.2]
    dashboard = MetricsDashboard(mv, port=8062, update_interval=200)
    fig = dashboard._build_figure(["loss"])
    assert len(fig.data) == 1
    assert fig.data[0].name == "Loss"


def test_cache_metrics_plot():
    mv = MetricsVisualizer()
    mv.metrics["cache_hit"] = [1, 2]
    dashboard = MetricsDashboard(mv, port=8063, update_interval=200)
    fig = dashboard._build_figure(["cache_hit"])
    assert len(fig.data) == 1
    assert fig.data[0].name == "Cache Hit"
