import time
import numpy as np
import torch
from sklearn.datasets import load_diabetes

from marble_core import Core, DataLoader
from marble_neuronenblitz import Neuronenblitz
from marble_brain import Brain
from marble_autograd import MarbleAutogradLayer
from tests.test_core_functions import minimal_params


def generate_dataset(n_samples: int = 50, seed: int = 0):
    """Generate simple (input, target) pairs using a sine function."""
    rng = np.random.default_rng(seed)
    xs = rng.uniform(-1.0, 1.0, size=n_samples)
    ys = np.sin(xs * np.pi)
    return list(zip(xs.tolist(), ys.tolist()))


def load_real_dataset(n_samples: int = 100):
    """Load a real regression dataset and return (input, target) pairs."""
    data = load_diabetes()
    xs = data.data[:n_samples, 0]
    ys = data.target[:n_samples]
    return list(zip(xs.astype(float).tolist(), ys.astype(float).tolist()))


def train_marble(train_data, val_data, epochs: int = 10):
    """Train using the pure MARBLE system."""
    params = minimal_params()
    core = Core(params)
    nb = Neuronenblitz(core)
    brain = Brain(core, nb, DataLoader())

    start = time.time()
    for _ in range(epochs):
        brain.train(train_data, epochs=1, validation_examples=val_data)
    val_loss = brain.validate(val_data)
    duration = time.time() - start
    return val_loss, duration


def train_autograd(train_data, val_data, epochs: int = 10, learning_rate: float = 0.01):
    """Train using the autograd pathway."""
    params = minimal_params()
    core = Core(params)
    nb = Neuronenblitz(core)
    brain = Brain(core, nb, DataLoader())
    layer = MarbleAutogradLayer(brain, learning_rate=learning_rate)

    start = time.time()
    for _ in range(epochs):
        for x, y in train_data:
            inp = torch.tensor(x, dtype=torch.float32, requires_grad=True)
            out = layer(inp)
            loss = (out - torch.tensor(y, dtype=torch.float32)) ** 2
            loss.backward()
    with torch.no_grad():
        errors = []
        for x, y in val_data:
            pred = layer(torch.tensor(x, dtype=torch.float32))
            errors.append(float(abs(y - pred.item())))
        val_loss = sum(errors) / len(errors) if errors else 0.0
    duration = time.time() - start
    return val_loss, duration


def run_benchmark():
    """Run both training modes and return their validation losses and durations."""
    data = load_real_dataset()
    train_data = data[:80]
    val_data = data[80:]

    marble_loss, marble_time = train_marble(train_data, val_data, epochs=10)
    autograd_loss, autograd_time = train_autograd(train_data, val_data, epochs=10)

    results = {
        "marble": {"loss": marble_loss, "time": marble_time},
        "autograd": {"loss": autograd_loss, "time": autograd_time},
    }
    return results


if __name__ == "__main__":
    res = run_benchmark()
    print("Benchmark results:")
    for k, v in res.items():
        print(f"{k}: loss={v['loss']:.4f}, time={v['time']:.2f}s")
