"""Download the Iris dataset and save as CSV.

This script fetches the Iris dataset from OpenML using scikit-learn and
stores it under the provided output path. The dataset is small and the
operation is identical on CPU or GPU systems.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from sklearn.datasets import fetch_openml


def download_iris(output: Path) -> None:
    """Fetch the Iris dataset and write it to ``output`` as CSV."""
    data = fetch_openml("iris", version=1, as_frame=True)
    df = data.frame
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the Iris dataset")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/iris.csv"),
        help="Destination file for the CSV output",
    )
    args = parser.parse_args()
    download_iris(args.output)


if __name__ == "__main__":
    main()
