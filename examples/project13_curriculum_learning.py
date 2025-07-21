import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from datasets import load_dataset
from tests.test_core_functions import minimal_params
from marble_core import Core
from marble_neuronenblitz import Neuronenblitz
from curriculum_learning import CurriculumLearner


def main() -> None:
    ds = load_dataset("mnist", split="train[:40]")
    pairs = [
        (
            float(record["image"].resize((8,8)).convert("L").getpixel((0,0)))/255.0,
            float(record["label"]),
        )
        for record in ds
    ]

    def difficulty(pair):
        return pair[1]

    core = Core(minimal_params())
    nb = Neuronenblitz(core)
    learner = CurriculumLearner(core, nb, difficulty_fn=difficulty)
    learner.train(pairs, epochs=2)
    print("History entries:", len(learner.history))


if __name__ == "__main__":
    main()
