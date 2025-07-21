import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from datasets import load_dataset
from pytorch_challenge import run_challenge


def main() -> None:
    load_dataset("mnist", split="train[:1]")
    res = run_challenge(seed=0)
    print(res)


if __name__ == "__main__":
    main()
