import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from datasets import load_dataset
from tests.test_core_functions import minimal_params
from marble_core import Core
from marble_neuronenblitz import Neuronenblitz
from transfer_learning import TransferLearner


def main() -> None:
    ds = load_dataset("mnist", split="train[:40]")
    pairs = [
        (
            float(record["image"].resize((8,8)).convert("L").getpixel((0,0)))/255.0,
            float(record["label"]),
        )
        for record in ds
    ]
    core = Core(minimal_params())
    nb = Neuronenblitz(core)
    learner = TransferLearner(core, nb, freeze_fraction=0.2)
    learner.train(pairs, epochs=1)
    print("Loss:", learner.history[-1]["loss"])


if __name__ == "__main__":
    main()
