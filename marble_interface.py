from __future__ import annotations

import pickle
import dill
import warnings
from typing import Any, Hashable, Iterable

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset

from huggingface_utils import hf_login
from marble import DataLoader

from autoencoder_learning import AutoencoderLearner
from config_loader import create_marble_from_config, load_config
from curriculum_learning import curriculum_train
from distillation_trainer import DistillationTrainer
from marble_autograd import MarbleAutogradLayer, TransparentMarbleLayer
from marble_main import MARBLE
from marble_brain import Brain
from marble_utils import core_from_json, core_to_json

warnings.filterwarnings(
    "ignore",
    message=r"`resume_download` is deprecated",
    category=FutureWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r"the load_module\(\) method is deprecated",
    category=DeprecationWarning,
)


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
    viz = getattr(marble, "metrics_visualizer", None)
    brain_viz = None
    if hasattr(marble, "brain"):
        brain_viz = getattr(marble.brain, "metrics_visualizer", None)
    fig = ax = ax_twin = csv_writer = json_writer = scheduler = None
    if viz is not None:
        fig = getattr(viz, "fig", None)
        ax = getattr(viz, "ax", None)
        viz.close()
        writer = None
        csv_writer = getattr(viz, "_csv_writer", None)
        json_writer = getattr(viz, "_json_writer", None)
        scheduler = getattr(viz, "backup_scheduler", None)
        ax_twin = getattr(viz, "ax_twin", None)
        viz.fig = None
        viz.ax = None
        if hasattr(viz, "ax_twin"):
            viz.ax_twin = None
        viz.writer = None
        viz._csv_writer = None
        viz._json_writer = None
        viz.backup_scheduler = None
    if brain_viz is not None:
        brain_viz.close()
        marble.brain.metrics_visualizer = None
    marble.metrics_visualizer = None
    with open(path, "wb") as f:
        dill.dump(marble, f)
    marble.metrics_visualizer = viz
    if brain_viz is not None:
        marble.brain.metrics_visualizer = brain_viz
    if viz is not None:
        viz.fig = fig
        viz.ax = ax
        if hasattr(viz, "ax_twin"):
            viz.ax_twin = ax_twin
        viz.writer = writer
        viz._csv_writer = csv_writer
        viz._json_writer = json_writer
        viz.backup_scheduler = scheduler


def load_marble_system(path: str) -> MARBLE:
    """Load a MARBLE system previously saved with :func:`save_marble_system`."""
    with open(path, "rb") as f:
        return dill.load(f)


def infer_marble_system(
    marble: MARBLE, input_value: float, *, tensor: bool = False
) -> Any:
    """Return model output for ``input_value`` using ``marble``.

    If ``tensor`` is ``True`` the raw encoded tensor is returned without
    decoding through the :class:`DataLoader` tokenizer.
    """
    return marble.get_brain().infer(input_value, tensor=tensor)


def train_marble_system(
    marble: MARBLE,
    train_examples: Iterable[Any],
    epochs: int = 1,
    validation_examples: Iterable[Any] | None = None,
    progress_callback=None,
) -> None:
    """Train ``marble`` on ``train_examples`` for ``epochs``."""
    marble.get_brain().train(
        train_examples,
        epochs=epochs,
        validation_examples=validation_examples,
        progress_callback=progress_callback,
    )


def distillation_train_marble_system(
    student: MARBLE,
    teacher: MARBLE,
    train_examples: Iterable[Any],
    epochs: int = 1,
    alpha: float = 0.5,
    validation_examples: Iterable[Any] | None = None,
) -> None:
    """Train ``student`` using ``teacher`` outputs for knowledge distillation."""
    trainer = DistillationTrainer(student.get_brain(), teacher.get_brain(), alpha=alpha)
    trainer.train(
        train_examples, epochs=epochs, validation_examples=validation_examples
    )


def curriculum_train_marble_system(
    marble: MARBLE,
    train_examples: Iterable[Any],
    *,
    epochs: int = 1,
    difficulty_fn=None,
    schedule: str = "linear",
) -> list[float]:
    """Train ``marble`` using curriculum learning."""

    core = marble.get_core()
    nb = marble.get_neuronenblitz()
    return curriculum_train(
        core,
        nb,
        list(train_examples),
        epochs=epochs,
        difficulty_fn=difficulty_fn,
        schedule=schedule,
    )


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

    warnings.filterwarnings(
        "ignore",
        message="`resume_download` is deprecated",
        category=FutureWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message="the load_module() method is deprecated",
        category=DeprecationWarning,
    )

    first_layer_size = None
    for layer in model.modules():
        if isinstance(layer, torch.nn.Linear):
            first_layer_size = layer.in_features
            break

    prediction_map = {}
    ds = load_dataset("mnist", split="test[:5]")
    with torch.no_grad():
        for record in ds:
            img = record["image"].convert("L")
            raw = torch.tensor(list(img.getdata()), dtype=torch.float32)
            flat = raw / 255.0
            out = model(flat.unsqueeze(0)).squeeze()
            mean_val = float(raw.mean())
            prediction_map[round(mean_val, 6)] = out.tolist()

    if brain_params is None:
        brain_params = {}
    brain_params = brain_params.copy()
    brain_params["pytorch_model"] = model
    brain_params["pytorch_input_size"] = first_layer_size
    brain_params["prediction_map"] = prediction_map

    return MARBLE(
        core_params or {},
        converter_model=model,
        nb_params=nb_params,
        brain_params=brain_params,
        dataloader_params=dataloader_params,
        init_from_weights=True,
    )


def load_hf_dataset(
    dataset_name: str,
    split: str,
    input_key: str = "input",
    target_key: str = "target",
    limit: int | None = None,
    streaming: bool = False,
    dataloader: "DataLoader | None" = None,
) -> list[tuple[Any, Any]]:
    """Load a Hugging Face dataset and return ``(input, target)`` pairs.

    When ``dataloader`` is provided, both inputs and targets are encoded
    using :class:`DataLoader.encode` to match Marble's training format.
    """
    token = hf_login()
    ds = load_dataset(dataset_name, split=split, token=token, streaming=streaming)
    examples: list[tuple[Any, Any]] = []
    for record in ds:
        inp = record[input_key]
        tgt = record[target_key]
        if dataloader is not None:
            inp = dataloader.encode(inp)
            tgt = dataloader.encode(tgt)
        examples.append((inp, tgt))
        if limit is not None and len(examples) >= limit:
            break
    return examples


def train_from_dataframe(marble: MARBLE, df: pd.DataFrame, epochs: int = 1) -> None:
    """Train ``marble`` using pairs from a :class:`pandas.DataFrame`."""
    examples = list(zip(df["input"].tolist(), df["target"].tolist()))
    train_marble_system(marble, examples, epochs=epochs)


def evaluate_marble_system(
    marble: MARBLE, examples: Iterable[tuple[Any, Any]]
) -> float:
    """Return mean squared error of ``marble`` predictions on ``examples``."""
    preds = []
    targets = []
    for inp, tgt in examples:
        preds.append(infer_marble_system(marble, inp))
        targets.append(tgt)
    arr_p = np.array(preds, dtype=float)
    arr_t = np.array(targets, dtype=float)
    return float(np.mean((arr_p - arr_t) ** 2))


def export_core_to_json(marble: MARBLE) -> str:
    """Serialize ``marble``\'s core to a JSON string."""
    return core_to_json(marble.get_core())


def import_core_from_json(
    json_str: str,
    nb_params: dict | None = None,
    brain_params: dict | None = None,
    dataloader_params: dict | None = None,
) -> MARBLE:
    """Create a new MARBLE instance from a core JSON string."""
    core = core_from_json(json_str)
    marble = MARBLE(
        core.params,
        nb_params=nb_params,
        brain_params=brain_params,
        dataloader_params=dataloader_params,
    )
    marble.core = core
    return marble


def save_core_json_file(marble: MARBLE, path: str) -> None:
    """Save the core of ``marble`` to ``path`` as JSON."""
    js = export_core_to_json(marble)
    with open(path, "w", encoding="utf-8") as f:
        f.write(js)


def load_core_json_file(
    path: str,
    nb_params: dict | None = None,
    brain_params: dict | None = None,
    dataloader_params: dict | None = None,
) -> MARBLE:
    """Load a MARBLE system from a core JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        js = f.read()
    return import_core_from_json(js, nb_params, brain_params, dataloader_params)


def add_neuron_to_marble(
    marble: MARBLE,
    neuron_type: str = "standard",
    tier: str | None = None,
) -> int:
    """Add a neuron to ``marble`` and return its ID."""
    core = marble.get_core()
    start = len(core.neurons)
    target = tier if tier is not None else core.choose_new_tier()
    core.expand(
        num_new_neurons=1,
        num_new_synapses=0,
        target_tier=target,
        neuron_types=neuron_type,
    )
    return start


def add_synapse_to_marble(
    marble: MARBLE,
    source_id: int,
    target_id: int,
    weight: float = 1.0,
    synapse_type: str = "standard",
) -> None:
    """Create a synapse in ``marble`` between two neurons."""
    marble.get_core().add_synapse(
        source_id, target_id, weight=weight, synapse_type=synapse_type
    )


def freeze_synapses_fraction(marble: MARBLE, fraction: float) -> None:
    """Freeze a fraction of synapses in ``marble``."""
    marble.get_core().freeze_fraction_of_synapses(fraction)


def expand_marble_core(
    marble: MARBLE,
    num_new_neurons: int = 10,
    num_new_synapses: int = 15,
    alternative_connection_prob: float = 0.1,
    target_tier: str | None = None,
    neuron_types: list[str] | str | None = None,
) -> None:
    """Expand the core of ``marble`` with new neurons and synapses."""
    marble.get_core().expand(
        num_new_neurons=num_new_neurons,
        num_new_synapses=num_new_synapses,
        alternative_connection_prob=alternative_connection_prob,
        target_tier=target_tier,
        neuron_types=neuron_types,
    )


def run_core_message_passing(marble: MARBLE, iterations: int = 1) -> float:
    """Run message passing on ``marble`` and return average change."""
    return marble.get_core().run_message_passing(iterations=iterations)


def increase_marble_representation(marble: MARBLE, delta: int = 1) -> None:
    """Increase representation size of ``marble``."""
    marble.get_core().increase_representation_size(delta)


def decrease_marble_representation(marble: MARBLE, delta: int = 1) -> None:
    """Decrease representation size of ``marble``."""
    marble.get_core().decrease_representation_size(delta)


def enable_marble_rl(marble: MARBLE) -> None:
    """Enable reinforcement learning inside ``marble``."""
    marble.get_core().enable_rl()


def disable_marble_rl(marble: MARBLE) -> None:
    """Disable reinforcement learning inside ``marble``."""
    marble.get_core().disable_rl()


def marble_select_action(marble: MARBLE, state: Hashable, n_actions: int) -> int:
    """Select an action using marble's Q-learning table."""
    return marble.get_core().rl_select_action(state, n_actions)


def marble_update_q(
    marble: MARBLE,
    state: Hashable,
    action: int,
    reward: float,
    next_state: Hashable,
    done: bool,
    n_actions: int = 4,
) -> None:
    """Update Q-table for marble's reinforcement learner."""
    marble.get_core().rl_update(state, action, reward, next_state, done, n_actions)


def cluster_marble_neurons(marble: MARBLE, k: int = 3) -> None:
    """Cluster marble's neurons into ``k`` groups."""
    marble.get_core().cluster_neurons(k)


def relocate_marble_clusters(
    marble: MARBLE, high: float = 1.0, medium: float = 0.1
) -> None:
    """Relocate neuron clusters based on attention scores."""
    marble.get_core().relocate_clusters(high, medium)


def extract_submarble(
    marble: MARBLE,
    neuron_ids: list[int],
    nb_params: dict | None = None,
    brain_params: dict | None = None,
    dataloader_params: dict | None = None,
) -> MARBLE:
    """Return a new MARBLE built from a subset of ``marble``'s neurons."""
    subcore = marble.get_core().extract_subcore(neuron_ids)
    new_marble = MARBLE(
        subcore.params,
        nb_params=nb_params,
        brain_params=brain_params,
        dataloader_params=dataloader_params,
    )
    new_marble.core = subcore
    return new_marble


def get_marble_status(marble: MARBLE) -> dict:
    """Return a detailed status dictionary of ``marble``'s core."""
    return marble.get_core().get_detailed_status()


def reset_core_representations(marble: MARBLE) -> None:
    """Reset all neuron representations in ``marble`` to zero."""
    for n in marble.get_core().neurons:
        n.representation[:] = 0


def randomize_core_representations(marble: MARBLE, std: float = 1.0) -> None:
    """Fill neuron representations with random values."""
    for n in marble.get_core().neurons:
        n.representation[:] = np.random.randn(*n.representation.shape) * std


def count_marble_synapses(marble: MARBLE) -> int:
    """Return the number of synapses in ``marble``."""
    return len(marble.get_core().synapses)


def train_autoencoder(
    marble: MARBLE,
    values: Iterable[float],
    epochs: int = 1,
    noise_std: float = 0.1,
    noise_decay: float = 0.99,
) -> float:
    """Train a denoising autoencoder and return the final loss."""

    learner = AutoencoderLearner(
        marble.get_core(),
        marble.get_neuronenblitz(),
        noise_std=float(noise_std),
        noise_decay=float(noise_decay),
    )
    learner.train(list(map(float, values)), epochs=int(epochs))
    return float(learner.history[-1]["loss"]) if learner.history else 0.0


def attach_marble_layer(
    model_or_path: str | torch.nn.Module,
    marble: MARBLE | Brain,
    after: int | str | None = None,
    train_in_graph: bool = True,
) -> torch.nn.Module:
    """Return ``model`` with an attached transparent Marble layer."""

    if isinstance(model_or_path, str):
        from torch_model_io import load_model_auto
        model = load_model_auto(model_or_path)
    else:
        model = model_or_path

    brain = marble.get_brain() if isinstance(marble, MARBLE) else marble
    hook = TransparentMarbleLayer(brain, train_in_graph=train_in_graph)

    if isinstance(model, torch.nn.Sequential):
        modules = list(model.children())
        idx = len(modules)
        if isinstance(after, int):
            idx = min(len(modules), after + 1)
        modules.insert(idx, hook)
        return torch.nn.Sequential(*modules)

    gm = torch.fx.symbolic_trace(model)
    name = f"marble_hook_{len(gm.graph.nodes)}"
    gm.add_submodule(name, hook)

    target = None
    if isinstance(after, str):
        for n in gm.graph.nodes:
            if n.op == "call_module" and n.target == after:
                target = n
                break
    elif isinstance(after, int):
        nodes = [n for n in gm.graph.nodes if n.op == "call_module"]
        if 0 <= after < len(nodes):
            target = nodes[after]
    if target is None:
        for n in reversed(gm.graph.nodes):
            if n.op == "call_module":
                target = n
                break

    with gm.graph.inserting_after(target):
        new_node = gm.graph.call_module(name, args=(target,))
    target.replace_all_uses_with(new_node)
    new_node.args = (target,)
    gm.graph.lint()
    gm.recompile()
    return gm


def save_attached_model(model: torch.nn.Module, path: str) -> None:
    """Persist ``model`` with attached MARBLE to ``path``."""
    from torch_model_io import save_entire_model
    save_entire_model(model, path)
