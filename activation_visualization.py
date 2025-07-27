import os
import numpy as np
import matplotlib.pyplot as plt
from marble_core import Core


def plot_activation_heatmap(core: Core, path: str) -> None:
    """Plot a heatmap of all neuron representations and save it to ``path``."""
    reps = np.stack([n.representation for n in core.neurons])
    plt.figure(figsize=(8, 6))
    plt.imshow(reps, aspect="auto", interpolation="nearest", cmap="viridis")
    plt.xlabel("Representation Index")
    plt.ylabel("Neuron ID")
    plt.colorbar(label="Activation")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
