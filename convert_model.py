import argparse
import torch
from pytorch_to_marble import convert_model
from marble_utils import core_to_json


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert PyTorch model checkpoint to MARBLE JSON"
    )
    parser.add_argument("--pytorch", required=True, help="Path to PyTorch model")
    parser.add_argument("--output", required=True, help="Output JSON path")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run conversion without saving JSON",
    )
    args = parser.parse_args()

    model = torch.load(args.pytorch, map_location="cpu")
    core = convert_model(model, dry_run=args.dry_run)
    if not args.dry_run:
        js = core_to_json(core)
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(js)


if __name__ == "__main__":
    main()

