import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from datasets import load_dataset
from tests.test_core_functions import minimal_params
from marble_core import Core
from marble_neuronenblitz import Neuronenblitz
from federated_learning import FederatedAveragingTrainer


def main() -> None:
    ds = load_dataset("mnist", split="train[:40]")
    pairs = [
        (
            float(record["image"].resize((8,8)).convert("L").getpixel((0,0)))/255.0,
            float(record["label"]),
        )
        for record in ds
    ]
    data1 = pairs[:20]
    data2 = pairs[20:]
    params = minimal_params()
    clients = []
    for _ in range(2):
        c = Core(params)
        nb = Neuronenblitz(c)
        clients.append((c, nb))
    trainer = FederatedAveragingTrainer(clients)
    trainer.train_round([data1, data2], epochs=1)
    print("Weights averaged across clients")


if __name__ == "__main__":
    main()
