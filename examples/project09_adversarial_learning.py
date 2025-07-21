import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from datasets import load_dataset
from tests.test_core_functions import minimal_params
from marble_core import Core
from marble_neuronenblitz import Neuronenblitz
from adversarial_learning import AdversarialLearner


def main() -> None:
    ds = load_dataset("mnist", split="train[:20]")
    real_values = [float(record["label"]) for record in ds]
    core = Core(minimal_params())
    gen = Neuronenblitz(core)
    disc = Neuronenblitz(core)
    learner = AdversarialLearner(core, gen, disc)
    learner.train(real_values, epochs=1)
    print("History length:", len(learner.history))


if __name__ == "__main__":
    main()
