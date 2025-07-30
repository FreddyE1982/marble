import argparse
from pathlib import Path

from pytorch_to_marble import convert_model
from marble_interface import MARBLE, save_marble_system
from marble_utils import core_to_json


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert PyTorch model checkpoint to MARBLE JSON or snapshot"
    )
    parser.add_argument("--pytorch", required=True, help="Path to PyTorch model")
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
    args = parser.parse_args()

    if not args.output and not (
        args.dry_run or args.summary or args.summary_output
    ):
        parser.error("--output is required unless running in dry-run or summary mode")

    from torch_model_io import load_model_auto
    model = load_model_auto(args.pytorch)

    if args.summary or args.summary_output:
        core, summary = convert_model(
            model, dry_run=True, return_summary=True
        )
        if args.summary_output:
            import json

            with open(args.summary_output, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)
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


if __name__ == "__main__":
    main()

