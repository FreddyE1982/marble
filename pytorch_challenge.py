import time
import random
from typing import List, Tuple, Dict

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.models import squeezenet1_1, SqueezeNet1_1_Weights
from sklearn.datasets import load_digits

from marble_core import Core, DataLoader
from marble_neuronenblitz import Neuronenblitz
from marble_brain import Brain
from tests.test_core_functions import minimal_params


def load_pretrained_model():
    """Return a pretrained small vision model."""
    model = squeezenet1_1(weights=SqueezeNet1_1_Weights.DEFAULT)
    model.eval()
    torch.set_num_threads(1)
    return model


def load_dataset(n_samples: int = 100) -> List[Tuple[np.ndarray, int]]:
    """Load digits dataset and return images and labels."""
    data = load_digits()
    imgs = data.images[:n_samples].astype(np.float32) / 16.0
    labels = data.target[:n_samples].astype(int)
    return list(zip(imgs, labels))


def _img_to_tensor(img: np.ndarray) -> torch.Tensor:
    t = torch.tensor(img).unsqueeze(0).unsqueeze(0)
    t = F.interpolate(t, size=(224, 224), mode="bilinear")
    t = t.repeat(1, 3, 1, 1)
    return t


def train_marble_with_challenge(
    train_data: List[Tuple[np.ndarray, int]],
    val_data: List[Tuple[np.ndarray, int]],
    pytorch_model: torch.nn.Module,
    epochs: int = 10,
    penalties: Dict[str, float] | None = None,
    seed: int | None = None,
) -> Dict[str, Dict[str, float]]:
    """Run challenge training and report final metrics."""
    if penalties is None:
        penalties = {"loss": 0.1, "speed": 0.1, "size": 0.1}
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    params = minimal_params()
    core = Core(params)
    nb = Neuronenblitz(core)
    brain = Brain(core, nb, DataLoader())

    marble_examples = [(float(img.mean()), float(label)) for img, label in train_data]
    torch_inputs = [_img_to_tensor(img) for img, _ in train_data]
    val_examples = [(float(img.mean()), float(lbl)) for img, lbl in val_data]
    brain.train_pytorch_challenge(
        marble_examples,
        pytorch_model,
        pytorch_inputs=torch_inputs,
        epochs=epochs,
        validation_examples=val_examples,
        loss_penalty=penalties["loss"],
        speed_penalty=penalties["speed"],
        size_penalty=penalties["size"],
    )

    marble_loss = brain.validate([(float(i.mean()), float(l)) for i, l in val_data])
    marble_size = (
        core.get_usage_by_tier("vram")
        + core.get_usage_by_tier("ram")
        + core.get_usage_by_tier("disk")
    )

    times = []
    for img, _ in val_data:
        start = time.time()
        _ = nb.dynamic_wander(float(img.mean()))
        times.append(time.time() - start)
    marble_time = sum(times) / len(times)

    pyro_times = []
    pyro_preds = []
    for img, _ in val_data:
        inp = _img_to_tensor(img)
        start = time.time()
        with torch.no_grad():
            out = pytorch_model(inp)
        pyro_times.append(time.time() - start)
        pyro_preds.append(int(out.argmax().item()))
    pyro_time = sum(pyro_times) / len(pyro_times)
    pyro_loss = float(
        np.mean([(lbl - pred) ** 2 for (_, lbl), pred in zip(val_data, pyro_preds)])
    )
    pyro_size = sum(p.numel() for p in pytorch_model.parameters()) * 4 / 1e6

    return {
        "marble": {"loss": marble_loss, "time": marble_time, "size": marble_size},
        "pytorch": {"loss": pyro_loss, "time": pyro_time, "size": pyro_size},
    }


def run_challenge(seed: int | None = None) -> Dict[str, Dict[str, float]]:
    data = load_dataset(100)
    train = data[:80]
    val = data[80:]
    model = load_pretrained_model()
    return train_marble_with_challenge(train, val, model, epochs=10, seed=seed)


if __name__ == "__main__":
    res = run_challenge()
    print(res)
