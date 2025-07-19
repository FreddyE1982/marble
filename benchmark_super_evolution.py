import time
from typing import List, Dict
from sklearn.datasets import load_diabetes

from marble_core import Core, DataLoader
from marble_neuronenblitz import Neuronenblitz
from marble_brain import Brain
from tests.test_core_functions import minimal_params


def load_dataset(n_samples: int = 100):
    data = load_diabetes()
    xs = data.data[:n_samples, 0]
    ys = data.target[:n_samples]
    return list(zip(xs.astype(float).tolist(), ys.astype(float).tolist()))


def train_super_evolution(train_data, val_data, epochs: int = 20, seed: int | None = None):
    params = minimal_params()
    if seed is not None:
        params["random_seed"] = seed
    core = Core(params)
    nb = Neuronenblitz(core)
    brain = Brain(core, nb, DataLoader(), super_evolution_mode=True)
    brain.core.cluster_neurons(k=3)
    brain.lobe_manager.organize()

    logs: List[Dict] = []
    change_idx = 0
    for _ in range(epochs):
        start = time.time()
        brain.train(train_data, epochs=1, validation_examples=val_data)
        val_loss = brain.validate(val_data)
        epoch_time = time.time() - start
        brain.super_evo_controller.record_metrics(val_loss, epoch_time)
        new_changes = brain.super_evo_controller.change_log[change_idx:]
        change_idx = len(brain.super_evo_controller.change_log)
        logs.append({"loss": float(val_loss), "time": epoch_time, "changes": new_changes})
    return logs


def run_benchmark(num_runs: int = 2, seed: int | None = None):
    data = load_dataset()
    train_data = data[:80]
    val_data = data[80:]
    results = []
    for _ in range(num_runs):
        logs = train_super_evolution(train_data, val_data, epochs=20, seed=seed)
        results.append(logs)
    return results


if __name__ == "__main__":
    out = run_benchmark()
    for i, run in enumerate(out):
        print(f"Run {i}")
        for e, log in enumerate(run, 1):
            print(f"Epoch {e}: loss={log['loss']:.4f}, time={log['time']:.2f}s")
            for ch in log["changes"]:
                print(f"  Changed {ch['parameter']} from {ch['old']:.4f} to {ch['new']:.4f}")

