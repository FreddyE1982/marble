from __future__ import annotations

import threading
from typing import Dict, Any

from dash import Dash, dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go


class MetricsDashboard:
    """Simple Plotly Dash dashboard to visualize metrics live."""

    def __init__(
        self,
        metrics_source: "MetricsVisualizer",
        host: str = "localhost",
        port: int = 8050,
        update_interval: int = 1000,
        window_size: int = 10,
    ) -> None:
        self.metrics_source = metrics_source
        self.host = host
        self.port = port
        self.update_interval = update_interval
        self.window_size = max(1, int(window_size))
        self.app = Dash(__name__)
        self.thread: threading.Thread | None = None
        self._setup_layout()

    def smooth(self, data):
        if len(data) <= self.window_size:
            return data
        result = []
        for i in range(len(data)):
            start = max(0, i - self.window_size + 1)
            result.append(sum(data[start : i + 1]) / (i - start + 1))
        return result

    def _setup_layout(self) -> None:
        metrics_keys = list(self.metrics_source.metrics.keys())
        self.app.layout = html.Div(
            [
                dcc.Graph(id="metrics-graph"),
                dcc.Checklist(
                    id="metric-select",
                    options=[{"label": k, "value": k} for k in metrics_keys],
                    value=metrics_keys,
                    inline=True,
                ),
                dcc.Interval(
                    id="interval", interval=self.update_interval, n_intervals=0
                ),
            ]
        )

        @self.app.callback(
            Output("metrics-graph", "figure"),
            Input("interval", "n_intervals"),
            Input("metric-select", "value"),
        )
        def update_graph(n: int, selected: list[str]) -> Dict[str, Any]:
            return self._build_figure(selected)

    def _build_figure(self, selected: list[str]) -> Dict[str, Any]:
        metrics = self.metrics_source.metrics
        fig = go.Figure()
        if "loss" in selected and metrics.get("loss"):
            fig.add_scatter(y=self.smooth(metrics["loss"]), mode="lines", name="Loss")
        if "vram_usage" in selected and metrics.get("vram_usage"):
            fig.add_scatter(
                y=self.smooth(metrics["vram_usage"]), mode="lines", name="VRAM Usage"
            )
        if "arousal" in selected and metrics.get("arousal"):
            fig.add_scatter(y=self.smooth(metrics["arousal"]), mode="lines", name="Arousal")
        if "stress" in selected and metrics.get("stress"):
            fig.add_scatter(y=self.smooth(metrics["stress"]), mode="lines", name="Stress")
        if "reward" in selected and metrics.get("reward"):
            fig.add_scatter(y=self.smooth(metrics["reward"]), mode="lines", name="Reward")
        if "plasticity_threshold" in selected and metrics.get("plasticity_threshold"):
            fig.add_scatter(
                y=self.smooth(metrics["plasticity_threshold"]), mode="lines", name="Plasticity"
            )
        if "message_passing_change" in selected and metrics.get("message_passing_change"):
            fig.add_scatter(
                y=self.smooth(metrics["message_passing_change"]), mode="lines", name="MsgPass"
            )
        if "compression_ratio" in selected and metrics.get("compression_ratio"):
            fig.add_scatter(
                y=self.smooth(metrics["compression_ratio"]), mode="lines", name="Compression"
            )
        if "meta_loss_avg" in selected and metrics.get("meta_loss_avg"):
            fig.add_scatter(
                y=self.smooth(metrics["meta_loss_avg"]), mode="lines", name="MetaLossAvg"
            )
        fig.update_layout(xaxis_title="Updates", yaxis_title="Value")
        return fig

    def start(self) -> None:
        if self.thread is None:
            self.thread = threading.Thread(
                target=self.app.run,
                kwargs={"host": self.host, "port": self.port, "debug": False},
                daemon=True,
            )
            self.thread.start()

    def stop(self) -> None:
        # Dash doesn't provide a direct stop method; thread ends when server stops
        pass
