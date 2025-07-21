import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from torchvision.datasets import STL10
from torchvision.transforms import ToTensor
from tests.test_core_functions import minimal_params
from marble_core import Core
from marble_neuronenblitz import Neuronenblitz
from contrastive_learning import ContrastiveLearner


def main() -> None:
    ds = STL10(root="data", split="unlabeled", download=True, transform=ToTensor())
    images = [float(img.mean().item()) for img, _ in ds[:20]]
    core = Core(minimal_params())
    nb = Neuronenblitz(core)
    learner = ContrastiveLearner(core, nb)
    loss = learner.train(images[:4])
    print("Loss:", loss)


if __name__ == "__main__":
    main()
