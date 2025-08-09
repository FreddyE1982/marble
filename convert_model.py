import argparse
from pathlib import Path
from typing import Dict

import yaml
import hashlib

from marble_interface import MARBLE, save_marble_system, load_marble_system
from marble_utils import core_to_json
from pytorch_to_marble import convert_model


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert PyTorch model checkpoint to MARBLE JSON or snapshot"
    )
    parser.add_argument("--pytorch", help="Path to PyTorch model")
    parser.add_argument(
        "--config", help="Path to YAML file providing default arguments"
    )
    parser.add_argument(
        "--output",
        help="Output path (.json or .marble)",
    )
    parser.add_argument(
        "--snapshot",
        action="store_true",
        help="Save converted model as a .marble snapshot. If --output is omitted,"
        " writes alongside the PyTorch file with a .marble extension",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run conversion without saving JSON",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print dry-run summary and exit",
    )
    parser.add_argument(
        "--summary-output",
        help="Path to save dry-run summary JSON",
    )
    parser.add_argument(
        "--summary-plot",
        help="Path to save bar chart of neurons and synapses per layer",
    )
    parser.add_argument(
        "--summary-csv",
        help="Path to save dry-run summary CSV",
    )
    parser.add_argument(
        "--summary-table",
        action="store_true",
        help="Print dry-run summary as a formatted table and exit",
    )
    parser.add_argument(
        "--summary-graph",
        help="Path to save dry-run graph HTML",
    )
    parser.add_argument(
        "--show-graph",
        action="store_true",
        help="Render converted graph in a web browser",
    )
    parser.add_argument(
        "--restore-hidden",
        action="store_true",
        help="Restore serialised RNN hidden states after conversion",
    )
    args = parser.parse_args()

    if args.config:
        with open(args.config, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        for key, value in cfg.items():
            if getattr(args, key, None) in (None, False):
                setattr(args, key, value)

    if not args.pytorch:
        parser.error("--pytorch is required (either via CLI or config file)")

    if not (args.output or args.snapshot) and not (
        args.dry_run
        or args.summary
        or args.summary_output
        or args.summary_plot
        or args.summary_csv
        or args.summary_table
        or args.summary_graph
        or args.show_graph
    ):
        parser.error(
            "--output is required unless running in dry-run or summary mode or --snapshot"
        )

    if args.snapshot:
        if args.output:
            args.output = str(Path(args.output).with_suffix(".marble"))
        else:
            args.output = str(Path(args.pytorch).with_suffix(".marble"))

    from torch_model_io import load_model_auto

    model = load_model_auto(args.pytorch)

    if (
        args.summary
        or args.summary_output
        or args.summary_plot
        or args.summary_csv
        or args.summary_table
        or args.summary_graph
    ):
        core, summary = convert_model(
            model, dry_run=True, return_summary=True, restore_hidden=args.restore_hidden
        )
        if args.summary_output:
            import json

            with open(args.summary_output, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)
        if args.summary_plot:
            _plot_summary(summary, args.summary_plot)
        if args.summary_csv:
            _summary_to_csv(summary, args.summary_csv)
        if args.summary_table:
            _summary_to_table(summary)
        if args.summary_graph:
            _graph_to_html(core, args.summary_graph)
        if args.show_graph:
            _show_graph(core)
        return

    core = convert_model(
        model, dry_run=args.dry_run, restore_hidden=args.restore_hidden
    )

    if args.show_graph:
        _show_graph(core)
    if args.dry_run or not args.output:
        return

    out_path = Path(args.output)
    core_json = core_to_json(core)
    if out_path.suffix == ".json":
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(core_json)
    elif out_path.suffix == ".marble":
        metadata = {
            "converter": "convert_model",
            "version": 1,
            "checksum": hashlib.sha256(core_json.encode("utf-8")).hexdigest(),
        }
        marble = MARBLE(core.params)
        marble.core = core
        marble.neuronenblitz.core = core
        marble.brain.core = core
        marble.metadata = metadata
        save_marble_system(marble, str(out_path))
        # Validate that the snapshot can be loaded back into a MARBLE system
        try:
            load_marble_system(str(out_path))
        except Exception as e:  # pragma: no cover - validation check
            raise RuntimeError(f"Saved snapshot failed to load: {e}") from e
    else:
        raise ValueError("Output extension must be .json or .marble")


def _plot_summary(summary: Dict[str, Dict], path: str) -> None:
    import matplotlib.pyplot as plt

    layers = list(summary["layers"].keys())
    neuron_counts = [info["neurons"] for info in summary["layers"].values()]
    synapse_counts = [info["synapses"] for info in summary["layers"].values()]
    x = range(len(layers))
    width = 0.35
    plt.figure()
    plt.bar(x, neuron_counts, width, label="neurons")
    plt.bar([i + width for i in x], synapse_counts, width, label="synapses")
    plt.xticks([i + width / 2 for i in x], layers, rotation=45, ha="right")
    plt.ylabel("Count")
    plt.title("Neurons and Synapses per Layer")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def _summary_to_csv(summary: Dict[str, Dict], path: str) -> None:
    import csv

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["layer", "neurons", "synapses"])
        for layer, info in summary["layers"].items():
            writer.writerow([layer, info["neurons"], info["synapses"]])


def _summary_to_table(summary: Dict[str, Dict]) -> None:
    """Print ``summary`` as a formatted table with device context."""
    import torch

    device = "GPU" if torch.cuda.is_available() else "CPU"
    print(f"Device: {device}")
    header = f"{'Layer':20} {'Neurons':>10} {'Synapses':>10}"
    print(header)
    print("-" * len(header))
    for layer, info in sorted(summary["layers"].items()):
        print(f"{layer:20} {info['neurons']:>10} {info['synapses']:>10}")
    print("-" * len(header))
    print(
        f"{'TOTAL':20} {summary['neurons']:>10} {summary['synapses']:>10}"
    )


def _graph_to_html(core, path: str) -> None:
    """Render ``core`` to an interactive HTML graph."""
    import networkx as nx
    import plotly.graph_objs as go
    from networkx_interop import core_to_networkx

    graph = core_to_networkx(core)
    pos = nx.spring_layout(graph, seed=42)
    edge_x = []
    edge_y = []
    for src, tgt in graph.edges():
        x0, y0 = pos[src]
        x1, y1 = pos[tgt]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]
    edge_trace = go.Scatter(x=edge_x, y=edge_y, mode="lines", line=dict(width=1))
    node_x = [pos[n][0] for n in graph.nodes()]
    node_y = [pos[n][1] for n in graph.nodes()]
    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers",
        marker=dict(size=5, color="#1f77b4"),
        text=list(graph.nodes()),
    )
    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(showlegend=False, xaxis=dict(visible=False), yaxis=dict(visible=False))
    fig.write_html(path)


def _show_graph(core) -> None:
    """Render ``core`` to an HTML file and open it in a browser."""
    from tempfile import NamedTemporaryFile
    import webbrowser

    with NamedTemporaryFile("w", delete=False, suffix=".html") as tmp:
        _graph_to_html(core, tmp.name)
        html_path = tmp.name
    print(f"Graph HTML saved to {html_path}")
    webbrowser.open(f"file://{html_path}")


if __name__ == "__main__":
    main()
