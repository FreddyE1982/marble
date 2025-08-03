import argparse
from pathlib import Path
from typing import Dict

import yaml

from marble_interface import MARBLE, save_marble_system
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
    args = parser.parse_args()

    if args.config:
        with open(args.config, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        for key, value in cfg.items():
            if getattr(args, key, None) in (None, False):
                setattr(args, key, value)

    if not args.pytorch:
        parser.error("--pytorch is required (either via CLI or config file)")

    if not args.output and not (
        args.dry_run or args.summary or args.summary_output or args.summary_plot
    ):
        parser.error("--output is required unless running in dry-run or summary mode")

    from torch_model_io import load_model_auto

    model = load_model_auto(args.pytorch)

    if args.summary or args.summary_output or args.summary_plot:
        core, summary = convert_model(model, dry_run=True, return_summary=True)
        if args.summary_output:
            import json

            with open(args.summary_output, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)
        if args.summary_plot:
            _plot_summary(summary, args.summary_plot)
        return

    core = convert_model(model, dry_run=args.dry_run)

    if args.dry_run:
        return

    out_path = Path(args.output)
    if out_path.suffix == ".json":
        js = core_to_json(core)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(js)
    elif out_path.suffix == ".marble":
        marble = MARBLE(core.params)
        marble.core = core
        marble.neuronenblitz.core = core
        marble.brain.core = core
        save_marble_system(marble, str(out_path))
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


if __name__ == "__main__":
    main()
