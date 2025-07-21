import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from datasets import load_dataset
from tests.test_core_functions import minimal_params
from marble_core import Core
from marble_neuronenblitz import Neuronenblitz
from semi_supervised_learning import SemiSupervisedLearner


def main() -> None:
    ds = load_dataset("mnist", split="train[:40]")
    pairs = [
        (
            float(record["image"].resize((8,8)).convert("L").getpixel((0,0)))/255.0,
            float(record["label"]),
        )
        for record in ds
    ]
    labeled = pairs[:20]
    unlabeled = [p[0] for p in pairs[20:]]
    core = Core(minimal_params())
    nb = Neuronenblitz(core)
    learner = SemiSupervisedLearner(core, nb)
    learner.train(labeled, unlabeled, epochs=1)
    print("Supervised loss:", learner.history[-1]["sup_loss"])


if __name__ == "__main__":
    main()
