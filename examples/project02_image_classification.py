import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
from datasets import load_dataset
from marble_main import MARBLE
from config_loader import load_config


def main() -> None:
    ds = load_dataset("cifar10", split="train[:200]")
    examples = [
        (float(np.array(record["img"]).mean()), float(record["label"]))
        for record in ds
    ]
    train = examples[:150]
    val = examples[150:]
    cfg = load_config()
    marble = MARBLE(cfg["core"])
    brain = marble.brain
    brain.start_training(train, epochs=1, validation_examples=val)
    brain.wait_for_training()
    print("Validation loss:", brain.validate(val))


if __name__ == "__main__":
    main()
