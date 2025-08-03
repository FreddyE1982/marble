import importlib

from marble_base import MetricsVisualizer
import global_workspace
from metrics_dashboard import MetricsDashboard


def test_workspace_metric_updates():
    importlib.reload(global_workspace)
    mv = MetricsVisualizer()
    gw = global_workspace.activate(capacity=2, metrics=mv)
    gw.publish("source", "msg")
    assert mv.metrics["workspace_queue"][-1] == 1
    dash = MetricsDashboard(mv)
    fig = dash._build_figure(["workspace_queue"])
    # Should produce a trace for workspace queue
    assert any(trace.name == "WorkspaceQueue" for trace in fig.data)
