import os
import numpy as np
import matplotlib.pyplot as plt
from marble_core import Core


def plot_activation_heatmap(core: Core, path: str, *, cmap: str = "viridis") -> None:
    """Plot a heatmap of all neuron representations and save it to ``path``.

    Parameters
    ----------
    core:
        The :class:`~marble_core.Core` instance whose neuron activations will
        be visualised.
    path:
        Destination filepath for the generated image.
    cmap:
        Matplotlib colour map to apply when rendering the heatmap. Examples are
        ``"viridis"`` (default), ``"plasma"`` or any name accepted by
        :func:`matplotlib.pyplot.colormaps`.
    """
    reps = np.stack([n.representation for n in core.neurons])
    plt.figure(figsize=(8, 6))
    plt.imshow(reps, aspect="auto", interpolation="nearest", cmap=cmap)
    plt.xlabel("Representation Index")
    plt.ylabel("Neuron ID")
    plt.colorbar(label="Activation")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
