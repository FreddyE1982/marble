import argparse

import torch

from bit_tensor_dataset import BitTensorDataset


def _load(path: str) -> BitTensorDataset:
    return torch.load(path, weights_only=False)


def _save(ds: BitTensorDataset, path: str) -> None:
    torch.save(ds, path)


def list_history(path: str) -> list[str]:
    ds = _load(path)
    return ds.history_ids()


def undo_cmd(path: str, steps: int, output: str) -> None:
    ds = _load(path)
    ds.undo(steps)
    _save(ds, output)


def redo_cmd(path: str, steps: int, output: str) -> None:
    ds = _load(path)
    ds.redo(steps)
    _save(ds, output)


def revert_cmd(path: str, snapshot_id: str, output: str) -> None:
    ds = _load(path)
    ds.revert_to(snapshot_id)
    _save(ds, output)


def _main() -> None:  # pragma: no cover - CLI entry
    parser = argparse.ArgumentParser(description="Dataset history management")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_hist = sub.add_parser("history", help="List snapshot IDs")
    p_hist.add_argument("dataset")

    p_undo = sub.add_parser("undo", help="Undo modifications")
    p_undo.add_argument("dataset")
    p_undo.add_argument("output")
    p_undo.add_argument("--steps", type=int, default=1)

    p_redo = sub.add_parser("redo", help="Redo modifications")
    p_redo.add_argument("dataset")
    p_redo.add_argument("output")
    p_redo.add_argument("--steps", type=int, default=1)

    p_rev = sub.add_parser("revert", help="Revert to a snapshot ID")
    p_rev.add_argument("dataset")
    p_rev.add_argument("snapshot_id")
    p_rev.add_argument("output")

    args = parser.parse_args()

    if args.cmd == "history":
        ids = list_history(args.dataset)
        print("\n".join(ids))
    elif args.cmd == "undo":
        undo_cmd(args.dataset, args.steps, args.output)
    elif args.cmd == "redo":
        redo_cmd(args.dataset, args.steps, args.output)
    elif args.cmd == "revert":
        revert_cmd(args.dataset, args.snapshot_id, args.output)
    else:  # pragma: no cover - defensive
        parser.error("Unknown command")


if __name__ == "__main__":  # pragma: no cover - CLI entry
    _main()
