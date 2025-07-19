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
    ) -> None:
        self.metrics_source = metrics_source
        self.host = host
        self.port = port
        self.update_interval = update_interval
        self.app = Dash(__name__)
        self.thread: threading.Thread | None = None
        self._setup_layout()

    def _setup_layout(self) -> None:
        self.app.layout = html.Div(
            [
                dcc.Graph(id="metrics-graph"),
                dcc.Interval(
                    id="interval", interval=self.update_interval, n_intervals=0
                ),
            ]
        )

        @self.app.callback(
            Output("metrics-graph", "figure"), Input("interval", "n_intervals")
        )
        def update_graph(n: int) -> Dict[str, Any]:
            metrics = self.metrics_source.metrics
            fig = go.Figure()
            if metrics.get("loss"):
                fig.add_scatter(y=metrics["loss"], mode="lines", name="Loss")
            if metrics.get("vram_usage"):
                fig.add_scatter(
                    y=metrics["vram_usage"], mode="lines", name="VRAM Usage"
                )
            if metrics.get("arousal"):
                fig.add_scatter(y=metrics["arousal"], mode="lines", name="Arousal")
            if metrics.get("stress"):
                fig.add_scatter(y=metrics["stress"], mode="lines", name="Stress")
            if metrics.get("reward"):
                fig.add_scatter(y=metrics["reward"], mode="lines", name="Reward")
            if metrics.get("plasticity_threshold"):
                fig.add_scatter(
                    y=metrics["plasticity_threshold"], mode="lines", name="Plasticity"
                )
            if metrics.get("message_passing_change"):
                fig.add_scatter(
                    y=metrics["message_passing_change"], mode="lines", name="MsgPass"
                )
            if metrics.get("compression_ratio"):
                fig.add_scatter(
                    y=metrics["compression_ratio"], mode="lines", name="Compression"
                )
            if metrics.get("meta_loss_avg"):
                fig.add_scatter(
                    y=metrics["meta_loss_avg"], mode="lines", name="MetaLossAvg"
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
