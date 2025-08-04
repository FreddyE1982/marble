import argparse
import torch
from config_loader import create_marble_from_config
from highlevel_pipeline import HighLevelPipeline


def _select_device(name: str) -> str:
    if name == "gpu":
        if torch.cuda.is_available():
            return "cuda"
        raise RuntimeError("GPU requested but CUDA is not available")
    return "cpu"


def _apply_device(pipeline: HighLevelPipeline, device: str) -> None:
    # override device for datasets created during execution
    pipeline.bit_dataset_params["device"] = device


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Utilities for HighLevelPipeline checkpoints"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    chk = sub.add_parser(
        "checkpoint", help="Execute pipeline and save a checkpoint afterwards"
    )
    chk.add_argument("pipeline", help="Path to pipeline JSON file")
    chk.add_argument("checkpoint", help="Output checkpoint file")
    chk.add_argument("--config", required=True, help="Path to MARBLE config YAML")
    chk.add_argument(
        "--device", choices=["cpu", "gpu"], default="cpu", help="Device to run on"
    )
    chk.add_argument(
        "--dataset-version",
        help="Optional dataset version metadata to store in the checkpoint",
    )

    res = sub.add_parser("resume", help="Resume pipeline from a checkpoint")
    res.add_argument("checkpoint", help="Path to checkpoint file")
    res.add_argument("--config", required=True, help="Path to MARBLE config YAML")
    res.add_argument(
        "--device", choices=["cpu", "gpu"], default="cpu", help="Device to run on"
    )

    args = parser.parse_args()
    device = _select_device(args.device)
    marble = create_marble_from_config(args.config)

    if args.command == "checkpoint":
        with open(args.pipeline, "r", encoding="utf-8") as f:
            pipe = HighLevelPipeline.load_json(f)
        if args.dataset_version:
            pipe.dataset_version = args.dataset_version
        _apply_device(pipe, device)
        pipe.execute(marble)
        pipe.save_checkpoint(args.checkpoint)
    else:  # resume
        pipe = HighLevelPipeline.load_checkpoint(args.checkpoint)
        _apply_device(pipe, device)
        pipe.execute(marble)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
