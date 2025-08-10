"""Download the IMDB sentiment analysis dataset.

This script retrieves the IMDB movie review dataset via the Hugging Face
``datasets`` library and saves the training and test splits as CSV files in the
specified output directory. The process behaves identically on CPU and GPU
systems.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from datasets import load_dataset


def download_imdb(output: Path) -> None:
    """Fetch the IMDB dataset and write train/test splits to CSV."""
    ds = load_dataset("imdb")
    output.mkdir(parents=True, exist_ok=True)
    ds["train"].to_pandas().to_csv(output / "train.csv", index=False)
    ds["test"].to_pandas().to_csv(output / "test.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the IMDB sentiment dataset")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/imdb"),
        help="Destination directory for the dataset",
    )
    args = parser.parse_args()
    download_imdb(args.output)


if __name__ == "__main__":
    main()
