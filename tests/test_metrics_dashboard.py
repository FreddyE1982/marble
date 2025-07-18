import time
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
