from __future__ import annotations

import pickle
from typing import Any, Iterable

from config_loader import create_marble_from_config, load_config
from marble_main import MARBLE
from marble_autograd import MarbleAutogradLayer


def new_marble_system(config_path: str | None = None) -> MARBLE:
    """Instantiate a :class:`MARBLE` system from an optional YAML config."""
    return create_marble_from_config(config_path)


def configure_marble_system(marble: MARBLE, config: str | dict) -> None:
    """Update an existing MARBLE system using a config path or dict."""
    cfg = load_config(config) if isinstance(config, str) else config
    core_params = cfg.get("core", {})
    nb_params = cfg.get("neuronenblitz", {})
    brain_params = cfg.get("brain", {})

    marble.core.params.update(core_params)
    for k, v in core_params.items():
        if hasattr(marble.core, k):
            setattr(marble.core, k, v)

    for k, v in nb_params.items():
        if hasattr(marble.neuronenblitz, k):
            setattr(marble.neuronenblitz, k, v)

    for k, v in brain_params.items():
        if hasattr(marble.brain, k):
            setattr(marble.brain, k, v)


def save_marble_system(marble: MARBLE, path: str) -> None:
    """Persist ``marble`` to ``path`` using pickle."""
    with open(path, "wb") as f:
        pickle.dump(marble, f)


def load_marble_system(path: str) -> MARBLE:
    """Load a MARBLE system previously saved with :func:`save_marble_system`."""
    with open(path, "rb") as f:
        return pickle.load(f)


def infer_marble_system(marble: MARBLE, input_value: float) -> float:
    """Return model output for ``input_value`` using ``marble``."""
    return marble.get_brain().infer(input_value)


def train_marble_system(
    marble: MARBLE,
    train_examples: Iterable[Any],
    epochs: int = 1,
    validation_examples: Iterable[Any] | None = None,
) -> None:
    """Train ``marble`` on ``train_examples`` for ``epochs``."""
    marble.get_brain().train(train_examples, epochs=epochs, validation_examples=validation_examples)


def set_dreaming(marble: MARBLE, enabled: bool) -> None:
    """Enable or disable dreaming for ``marble``."""
    if enabled:
        marble.get_brain().start_dreaming()
    else:
        marble.get_brain().stop_dreaming()


def set_autograd(marble: MARBLE, enabled: bool, learning_rate: float = 0.01) -> None:
    """Toggle the autograd layer on ``marble``."""
    if enabled and marble.get_autograd_layer() is None:
        layer = MarbleAutogradLayer(marble.get_brain(), learning_rate=learning_rate)
        marble.get_brain().set_autograd_layer(layer)
        marble.autograd_layer = layer
    elif not enabled and marble.get_autograd_layer() is not None:
        marble.get_brain().set_autograd_layer(None)
        marble.autograd_layer = None


def convert_pytorch_model(
    model: "torch.nn.Module",
    core_params: dict | None = None,
    nb_params: dict | None = None,
    brain_params: dict | None = None,
    dataloader_params: dict | None = None,
) -> MARBLE:
    """Return a :class:`MARBLE` instance converted from a PyTorch ``model``.

    Parameters
    ----------
    model:
        The PyTorch model to convert.
    core_params:
        Optional parameters passed to :class:`MARBLE` for core creation.
    nb_params:
        Optional :class:`Neuronenblitz` parameters.
    brain_params:
        Optional :class:`Brain` parameters.
    dataloader_params:
        Optional :class:`DataLoader` parameters.

    Returns
    -------
    MARBLE
        A new MARBLE system initialized from ``model`` weights.
    """

    return MARBLE(
        core_params or {},
        converter_model=model,
        nb_params=nb_params,
        brain_params=brain_params,
        dataloader_params=dataloader_params,
        init_from_weights=True,
    )
