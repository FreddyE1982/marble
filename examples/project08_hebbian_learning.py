import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from datasets import load_dataset
from tests.test_core_functions import minimal_params
from marble_core import Core
from marble_neuronenblitz import Neuronenblitz
from hebbian_learning import HebbianLearner


def main() -> None:
    ds = load_dataset("mnist", split="train[:20]")
    inputs = [float(record["image"].resize((8,8)).convert("L").getpixel((0,0)))/255.0 for record in ds]
    core = Core(minimal_params())
    nb = Neuronenblitz(core)
    learner = HebbianLearner(core, nb)
    learner.train(inputs, epochs=1)
    print("Trained on", len(learner.history), "samples")


if __name__ == "__main__":
    main()
