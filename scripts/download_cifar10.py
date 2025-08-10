"""Download the CIFAR-10 dataset.

This script fetches the CIFAR-10 image classification dataset using
``torchvision`` and stores the training and test splits under the provided
output directory. The dataset is identical on CPU and GPU systems.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from torchvision.datasets import CIFAR10


def download_cifar10(output: Path) -> None:
    """Download the CIFAR-10 dataset to ``output``.

    Both the training and test splits are fetched so subsequent experiments can
    access the full dataset without additional downloads.
    """
    output.mkdir(parents=True, exist_ok=True)
    CIFAR10(root=output, train=True, download=True)
    CIFAR10(root=output, train=False, download=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the CIFAR-10 dataset")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/cifar10"),
        help="Destination directory for the dataset",
    )
    args = parser.parse_args()
    download_cifar10(args.output)


if __name__ == "__main__":
    main()
