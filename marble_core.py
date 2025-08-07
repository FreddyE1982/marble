# ruff: noqa
from __future__ import annotations

import contextlib
import copy
import os
import pickle
import random
import time
from collections import deque
from datetime import datetime
from typing import Any, Callable, Hashable, Type

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from tokenizers import Tokenizer

import tensor_backend as tb
from core.init_seed import compute_mandelbrot, generate_seed
from core.message_passing import AttentionModule, perform_message_passing
from event_bus import global_event_bus
from graph_cache import GRAPH_CACHE, GraphKey
from marble_base import MetricsVisualizer
from marble_imports import *  # noqa: F401,F403,F405
from memory_pool import MemoryPool


def init_distributed(world_size: int, rank: int = 0, backend: str = "gloo") -> bool:
    """Initialize torch distributed if available.

    Parameters
    ----------
    world_size:
        Total number of processes participating in training.
    rank:
        Rank of the current process.
    backend:
        Backend to use (``gloo`` or ``nccl``).

    Returns
    -------
    bool
        ``True`` when initialization succeeded, ``False`` otherwise.
    """

    if not dist.is_available():  # pragma: no cover - depends on torch build
        return False
    if dist.is_initialized():  # pragma: no cover - already init
        return True
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    dist.init_process_group(backend=backend, world_size=world_size, rank=rank)
    return True


def cleanup_distributed() -> None:
    """Destroy the default distributed process group if initialized."""

    if dist.is_available() and dist.is_initialized():  # pragma: no cover
        dist.destroy_process_group()


# Representation size for GNN-style message passing
_REP_SIZE = 4


def _neuron_factory() -> "Neuron":
    """Factory for creating blank neurons used by :class:`MemoryPool`."""
    return Neuron(-1, rep_size=_REP_SIZE)


def _synapse_factory() -> "Synapse":
    """Factory for creating blank synapses used by :class:`MemoryPool`."""
    return Synapse(0, 0)


def _init_weights(
    rep_size: int,
    *,
    strategy: str = "uniform",
    mean: float = 0.0,
    std: float = 1.0,
    min_val: float = -0.1,
    max_val: float = 0.1,
):
    """Initialise MLP weights using the chosen strategy."""
    rs = np.random.RandomState(0)
    if strategy == "normal":
        w1 = rs.normal(mean, std, size=(rep_size, 8))
        b1 = rs.normal(mean, std, size=8)
        w2 = rs.normal(mean, std, size=(8, rep_size))
        b2 = rs.normal(mean, std, size=rep_size)
    else:
        w1 = rs.uniform(min_val, max_val, size=(rep_size, 8))
        b1 = rs.uniform(min_val, max_val, size=8)
        w2 = rs.uniform(min_val, max_val, size=(8, rep_size))
        b2 = rs.uniform(min_val, max_val, size=rep_size)
    return w1, b1, w2, b2


_W1, _B1, _W2, _B2 = _init_weights(_REP_SIZE)


def configure_representation_size(
    rep_size: int,
    *,
    weight_strategy: str = "uniform",
    mean: float = 0.0,
    std: float = 1.0,
    min_val: float = -0.1,
    max_val: float = 0.1,
) -> None:
    """Configure global representation size used for message passing."""
    global _REP_SIZE, _W1, _B1, _W2, _B2
    _REP_SIZE = rep_size
    _W1, _B1, _W2, _B2 = _init_weights(
        rep_size,
        strategy=weight_strategy,
        mean=mean,
        std=std,
        min_val=min_val,
        max_val=max_val,
    )


def resize_neuron_representations(core: "Core", new_size: int) -> None:
    """Resize all neuron representations in ``core`` to ``new_size``.

    This updates the global representation configuration and preserves existing
    values by truncation or zero-padding as needed.
    """
    old_size = core.rep_size
    if new_size == old_size:
        return
    configure_representation_size(new_size)
    for neuron in core.neurons:
        rep = neuron.representation
        if new_size > rep.size:
            padded = np.zeros(new_size, dtype=rep.dtype)
            padded[: rep.size] = rep
            neuron.representation = padded
        else:
            neuron.representation = rep[:new_size]


def _apply_activation(arr: np.ndarray, activation: str) -> np.ndarray:
    """Return ``arr`` passed through the given activation function."""
    if activation == "relu":
        return tb.relu(arr)
    if activation == "sigmoid":
        return tb.sigmoid(arr)
    return np.tanh(arr)


def _layer_norm(arr: np.ndarray) -> np.ndarray:
    """Return array normalized to zero mean and unit variance."""
    mean = arr.mean(axis=-1, keepdims=True)
    var = arr.var(axis=-1, keepdims=True) + 1e-6
    return (arr - mean) / np.sqrt(var)


def _apply_activation_torch(arr: "torch.Tensor", activation: str) -> "torch.Tensor":
    if activation == "relu":
        return torch.relu(arr)
    if activation == "sigmoid":
        return torch.sigmoid(arr)
    return torch.tanh(arr)


def _simple_mlp_impl(
    x: np.ndarray | "torch.Tensor",
    activation: str = "tanh",
    *,
    apply_layer_norm: bool = True,
    mixed_precision: bool = False,
) -> np.ndarray | "torch.Tensor":
    """Implementation of the tiny MLP used throughout MARBLE."""

    if torch.is_tensor(x):
        device = x.device
        autocast = (
            torch.autocast(device_type=device.type)
            if mixed_precision
            else contextlib.nullcontext()
        )
        with autocast:
            w1 = torch.as_tensor(_W1, device=device, dtype=x.dtype)
            b1 = torch.as_tensor(_B1, device=device, dtype=x.dtype)
            h = _apply_activation_torch(x @ w1 + b1, activation)
            if apply_layer_norm:
                h = torch.nn.functional.layer_norm(h, h.shape[-1:])
            w2 = torch.as_tensor(_W2, device=device, dtype=x.dtype)
            b2 = torch.as_tensor(_B2, device=device, dtype=x.dtype)
            out = _apply_activation_torch(h @ w2 + b2, activation)
        if not torch.all(torch.isfinite(out)):
            raise ValueError("NaN or Inf encountered in MLP output")
        return out
    # Handle potential NaNs or infinities to avoid runtime warnings
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    h = _apply_activation(tb.matmul(x, _W1) + _B1, activation)
    if apply_layer_norm:
        h = _layer_norm(h)
    out = _apply_activation(tb.matmul(h, _W2) + _B2, activation)
    if not np.all(np.isfinite(out)):
        raise ValueError("NaN or Inf encountered in MLP output")
    return out


def _simple_mlp(
    x: np.ndarray | "torch.Tensor",
    activation: str = "tanh",
    *,
    apply_layer_norm: bool = True,
    mixed_precision: bool = False,
) -> np.ndarray | "torch.Tensor":
    """Tiny MLP with one hidden layer and configurable activations.

    When graph precompilation is enabled the Torch execution graph for the
    given input signature is cached and reused across calls.  The cache key
    incorporates shape, dtype, device and relevant keyword arguments.
    """

    if torch.is_tensor(x) and GRAPH_CACHE.enabled:
        key = GraphKey(
            "simple_mlp",
            tuple(x.shape),
            x.dtype,
            x.device,
            extras=(activation, apply_layer_norm, mixed_precision),
        )

        def fn(inp: "torch.Tensor") -> "torch.Tensor":
            return _simple_mlp_impl(
                inp,
                activation,
                apply_layer_norm=apply_layer_norm,
                mixed_precision=mixed_precision,
            )

        compiled = GRAPH_CACHE.precompile(key, fn, x)
        return compiled(x)

    return _simple_mlp_impl(
        x,
        activation,
        apply_layer_norm=apply_layer_norm,
        mixed_precision=mixed_precision,
    )


def precompile_simple_mlp(
    example: "torch.Tensor",
    activation: str = "tanh",
    *,
    apply_layer_norm: bool = True,
    mixed_precision: bool = False,
) -> None:
    """Precompile the simple MLP for a representative ``example`` tensor."""

    GRAPH_CACHE.enable(True)
    key = GraphKey(
        "simple_mlp",
        tuple(example.shape),
        example.dtype,
        example.device,
        extras=(activation, apply_layer_norm, mixed_precision),
    )

    def fn(inp: "torch.Tensor") -> "torch.Tensor":
        return _simple_mlp_impl(
            inp,
            activation,
            apply_layer_norm=apply_layer_norm,
            mixed_precision=mixed_precision,
        )

    GRAPH_CACHE.precompile(key, fn, example)


# List of supported neuron types including layer-mimicking variants
NEURON_TYPES = [
    "standard",
    "excitatory",
    "inhibitory",
    "modulatory",
    "linear",
    "conv1d",
    "conv2d",
    "batchnorm",
    "dropout",
    "relu",
    "leakyrelu",
    "elu",
    "sigmoid",
    "tanh",
    "gelu",
    "softmax",
    "maxpool1d",
    "avgpool1d",
    "flatten",
    "convtranspose1d",
    "convtranspose2d",
    "lstm",
    "gru",
    "layernorm",
    "conv3d",
    "convtranspose3d",
    "maxpool2d",
    "avgpool2d",
    "dropout2d",
    "prelu",
    "embedding",
    "rnn",
]

# List of supported synapse types
SYNAPSE_TYPES = [
    "standard",
    "one_way",
    "mirror",
    "multi_neuron",
    "recurrent",
    "excitatory",
    "inhibitory",
    "modulatory",
    "interconnection",
    "dropout",
    "batchnorm",
]

# Registry of custom loss modules mapped by name
LOSS_MODULES: dict[str, Type[nn.Module]] = {}

# Global registry for all tiers
TIER_REGISTRY = {}


class TierMeta(type):
    def __init__(cls, name, bases, attrs):
        super().__init__(name, bases, attrs)
        if name != "Tier":
            instance = cls()
            TIER_REGISTRY[instance.name] = instance
            print(f"Tier '{instance.name}' registered automatically.")


class Tier(metaclass=TierMeta):
    def __init__(self):
        self.name = self.__class__.__name__
        self.description = "Generic tier functionality."
        self.limit_mb = None
        self.order = 100

    def process(self, data: Any) -> Any:
        return data


class VramTier(Tier):
    def __init__(self):
        super().__init__()
        self.name = "vram"
        self.description = "Fast memory (VRAM) tier."
        self.limit_mb = 100


class RamTier(Tier):
    def __init__(self):
        super().__init__()
        self.name = "ram"
        self.description = "Intermediate memory (RAM) tier."
        self.limit_mb = 500


class DiskTier(Tier):
    def __init__(self):
        super().__init__()
        self.name = "disk"
        self.description = "Persistent memory (Disk) tier."
        self.limit_mb = 10000


class FileTier(Tier):
    def __init__(self):
        super().__init__()
        self.name = "file"
        self.description = "A tier that writes modified data into a designated file."
        self.limit_mb = 50
        self.file_path = os.path.join("data", "marble_file_tier.dat")
        os.makedirs(os.path.dirname(self.file_path), exist_ok=True)
        if not os.path.exists(self.file_path):
            with open(self.file_path, "wb") as f:
                f.write(b"")

    def process(self, data: Any) -> Any:
        modified_data = data * 1.1
        with open(self.file_path, "ab") as f:
            f.write(pickle.dumps(modified_data))
        return modified_data


class RemoteTier(Tier):
    def __init__(self):
        super().__init__()
        self.name = "remote"
        self.description = "Remote compute tier accessed via HTTP."
        self.limit_mb = None


class InvalidNeuronParamsError(ValueError):
    """Raised when neuron parameters fail validation."""


class Neuron:
    def __init__(
        self,
        nid: Hashable,
        value: float = 0.0,
        tier: str = "vram",
        neuron_type: str = "standard",
        rep_size: int = _REP_SIZE,
    ) -> None:
        self.id = nid
        self.value = value
        if tier not in TIER_REGISTRY:
            raise InvalidNeuronParamsError(
                f"Unknown tier '{tier}'. Valid options are: {', '.join(TIER_REGISTRY)}"
            )
        self.tier = tier
        if neuron_type not in NEURON_TYPES:
            raise InvalidNeuronParamsError(
                f"Unknown neuron_type '{neuron_type}'. Valid options are: {', '.join(NEURON_TYPES)}"
            )
        self.neuron_type = neuron_type
        self.synapses = []
        self.formula = None
        self.created_at = datetime.now()
        self.cluster_id = None
        self.attention_score = 0.0
        self.energy = 1.0
        if rep_size <= 0:
            raise InvalidNeuronParamsError("rep_size must be positive")
        self.representation = np.zeros(rep_size, dtype=float)
        self.params = {}
        # Flag used during message passing to indicate active neurons.
        # Stored in params so converters and builders can toggle it.
        self.params["activation_flag"] = False
        self.value_history = []
        self.initialize_params()
        self.validate_params()

    def validate_params(self) -> None:
        """Validate the parameters stored in ``self.params``."""
        if "stride" in self.params and (
            not isinstance(self.params["stride"], int) or self.params["stride"] <= 0
        ):
            raise InvalidNeuronParamsError("stride must be a positive integer")
        if "p" in self.params:
            p = float(self.params["p"])
            if not 0.0 <= p <= 1.0:
                raise InvalidNeuronParamsError(
                    "dropout probability must be between 0 and 1"
                )
        if "kernel" in self.params and not isinstance(
            self.params["kernel"], np.ndarray
        ):
            raise InvalidNeuronParamsError("kernel must be a numpy.ndarray")
        if "kernel" in self.params:
            k = self.params["kernel"]
            if self.neuron_type in {"conv1d", "convtranspose1d"} and k.ndim != 1:
                raise InvalidNeuronParamsError("conv1d kernel must be 1D")
            if self.neuron_type in {"conv2d", "convtranspose2d"} and k.ndim != 2:
                raise InvalidNeuronParamsError("conv2d kernel must be 2D")
            if self.neuron_type in {"conv3d", "convtranspose3d"} and k.ndim != 3:
                raise InvalidNeuronParamsError("conv3d kernel must be 3D")
        if "padding" in self.params and (
            not isinstance(self.params["padding"], int) or self.params["padding"] < 0
        ):
            raise InvalidNeuronParamsError("padding must be a non-negative integer")
        if "output_padding" in self.params:
            op = self.params["output_padding"]
            stride = self.params.get("stride", 1)
            if (
                not isinstance(op, int)
                or op < 0
                or (isinstance(stride, int) and op >= stride)
            ):
                raise InvalidNeuronParamsError(
                    "output_padding must be >=0 and less than stride"
                )
        if "negative_slope" in self.params and self.params["negative_slope"] < 0:
            raise InvalidNeuronParamsError("negative_slope must be non-negative")
        if "alpha" in self.params and self.params["alpha"] <= 0:
            raise InvalidNeuronParamsError("alpha must be positive")
        if "momentum" in self.params:
            mom = float(self.params["momentum"])
            if not 0.0 < mom < 1.0:
                raise InvalidNeuronParamsError("momentum must be between 0 and 1")
        if "eps" in self.params and self.params["eps"] <= 0:
            raise InvalidNeuronParamsError("eps must be positive")
        if "size" in self.params and (
            not isinstance(self.params["size"], int) or self.params["size"] <= 0
        ):
            raise InvalidNeuronParamsError("size must be a positive integer")
        if "axis" in self.params and not isinstance(self.params["axis"], int):
            raise InvalidNeuronParamsError("axis must be an integer")
        if {
            "num_embeddings",
            "embedding_dim",
        }.issubset(self.params):
            num = self.params["num_embeddings"]
            dim = self.params["embedding_dim"]
            weights = self.params.get("weights")
            if not isinstance(num, int) or num <= 0:
                raise InvalidNeuronParamsError(
                    "num_embeddings must be a positive integer"
                )
            if not isinstance(dim, int) or dim <= 0:
                raise InvalidNeuronParamsError(
                    "embedding_dim must be a positive integer"
                )
            if weights is not None and (
                not isinstance(weights, np.ndarray) or weights.shape != (num, dim)
            ):
                raise InvalidNeuronParamsError(
                    "weights shape must match (num_embeddings, embedding_dim)"
                )

    def initialize_params(self) -> None:
        """Initialize layer-like parameters based on ``neuron_type``."""
        if self.neuron_type == "linear":
            self.params = {
                "weight": random.uniform(-1.0, 1.0),
                "bias": random.uniform(-1.0, 1.0),
            }
        elif self.neuron_type == "conv1d":
            kernel = np.random.randn(3).astype(float)
            self.params = {"kernel": kernel, "stride": 1}
        elif self.neuron_type == "conv2d":
            kernel = np.random.randn(3, 3).astype(float)
            self.params = {"kernel": kernel, "stride": 1, "padding": 0}
        elif self.neuron_type == "batchnorm":
            self.params = {
                "mean": 0.0,
                "var": 1.0,
                "momentum": 0.1,
                "eps": 1e-5,
            }
        elif self.neuron_type == "dropout":
            self.params = {"p": 0.5}
        elif self.neuron_type == "leakyrelu":
            self.params = {"negative_slope": 0.01}
        elif self.neuron_type == "elu":
            self.params = {"alpha": 1.0}
        elif self.neuron_type == "softmax":
            self.params = {"axis": -1}
        elif self.neuron_type in {"relu", "sigmoid", "tanh", "gelu", "flatten"}:
            self.params = {}
        elif self.neuron_type in {"maxpool1d", "avgpool1d"}:
            self.params = {"size": 2, "stride": 2}
        elif self.neuron_type == "convtranspose1d":
            kernel = np.random.randn(3).astype(float)
            self.params = {
                "kernel": kernel,
                "stride": 1,
                "padding": 0,
                "output_padding": 0,
            }
        elif self.neuron_type == "convtranspose2d":
            kernel = np.random.randn(3, 3).astype(float)
            self.params = {
                "kernel": kernel,
                "stride": 1,
                "padding": 0,
                "output_padding": 0,
            }
        elif self.neuron_type == "conv3d":
            kernel = np.random.randn(3, 3, 3).astype(float)
            self.params = {"kernel": kernel, "stride": 1, "padding": 0}
        elif self.neuron_type == "convtranspose3d":
            kernel = np.random.randn(3, 3, 3).astype(float)
            self.params = {
                "kernel": kernel,
                "stride": 1,
                "padding": 0,
                "output_padding": 0,
            }
        elif self.neuron_type in {"maxpool2d", "avgpool2d"}:
            self.params = {"size": 2, "stride": 2}
        elif self.neuron_type == "dropout2d":
            self.params = {"p": 0.5}
        elif self.neuron_type == "prelu":
            self.params = {"alpha": 0.25}
        elif self.neuron_type == "embedding":
            num = 10
            dim = 4
            self.params = {
                "num_embeddings": num,
                "embedding_dim": dim,
                "weights": np.random.randn(num, dim).astype(float),
            }
        elif self.neuron_type == "rnn":
            self.params = {
                "wx": random.uniform(-1.0, 1.0),
                "wh": random.uniform(-1.0, 1.0),
                "b": 0.0,
            }
            self.hidden_state = 0.0
        elif self.neuron_type == "lstm":
            self.params = {
                "wi": random.uniform(-1.0, 1.0),
                "ui": random.uniform(-1.0, 1.0),
                "bi": 0.0,
                "wf": random.uniform(-1.0, 1.0),
                "uf": random.uniform(-1.0, 1.0),
                "bf": 0.0,
                "wo": random.uniform(-1.0, 1.0),
                "uo": random.uniform(-1.0, 1.0),
                "bo": 0.0,
                "wg": random.uniform(-1.0, 1.0),
                "ug": random.uniform(-1.0, 1.0),
                "bg": 0.0,
            }
            self.hidden_state = 0.0
            self.cell_state = 0.0
        elif self.neuron_type == "gru":
            self.params = {
                "wr": random.uniform(-1.0, 1.0),
                "ur": random.uniform(-1.0, 1.0),
                "br": 0.0,
                "wz": random.uniform(-1.0, 1.0),
                "uz": random.uniform(-1.0, 1.0),
                "bz": 0.0,
                "wn": random.uniform(-1.0, 1.0),
                "un": random.uniform(-1.0, 1.0),
                "bn": 0.0,
            }
            self.hidden_state = 0.0
        elif self.neuron_type == "layernorm":
            self.params = {"eps": 1e-5, "gain": 1.0, "bias": 0.0, "axis": -1}

    def process(self, value: float) -> float:
        """Apply the neuron-type specific transformation to ``value``."""
        if self.neuron_type == "linear":
            w = self.params.get("weight", 1.0)
            b = self.params.get("bias", 0.0)
            return w * value + b
        elif self.neuron_type == "conv1d":
            kernel = self.params.get("kernel", np.array([1.0]))
            stride = self.params.get("stride", 1)
            self.value_history.append(value)
            if len(self.value_history) < len(kernel):
                padded = [0.0] * (
                    len(kernel) - len(self.value_history)
                ) + self.value_history
            else:
                padded = self.value_history[-len(kernel) :]
                if len(self.value_history) > len(kernel):
                    self.value_history = self.value_history[stride:]
            flipped = kernel[::-1]
            conv = sum(flipped[i] * padded[i] for i in range(len(kernel)))
            return float(conv)
        elif self.neuron_type == "conv2d":
            kernel = self.params.get("kernel", np.ones((1, 1)))
            stride = self.params.get("stride", 1)
            padding = self.params.get("padding", 0)
            if torch.is_tensor(value):
                arr = value.detach().cpu().numpy()
                use_torch = True
            else:
                arr = cp.asarray(value)
                use_torch = False
            k = cp.asarray(kernel)
            if padding > 0:
                arr = cp.pad(
                    arr,
                    ((padding, padding), (padding, padding)),
                    mode="constant",
                )
            out_h = (arr.shape[0] - k.shape[0]) // stride + 1
            out_w = (arr.shape[1] - k.shape[1]) // stride + 1
            result = cp.zeros((out_h, out_w), dtype=arr.dtype)
            for i in range(out_h):
                for j in range(out_w):
                    region = arr[
                        i * stride : i * stride + k.shape[0],
                        j * stride : j * stride + k.shape[1],
                    ]
                    result[i, j] = cp.sum(region * k)
            if use_torch:
                return torch.from_numpy(cp.asnumpy(result))
            if isinstance(value, np.ndarray) and not CUDA_AVAILABLE:
                return cp.asnumpy(result)
            return result
        elif self.neuron_type == "convtranspose2d":
            kernel = self.params.get("kernel", np.ones((1, 1)))
            stride = self.params.get("stride", 1)
            padding = self.params.get("padding", 0)
            out_pad = self.params.get("output_padding", 0)
            if torch.is_tensor(value):
                arr = value.detach().cpu().numpy()
                use_torch = True
            else:
                arr = cp.asarray(value)
                use_torch = False
            k = cp.asarray(kernel)
            out_h = (arr.shape[0] - 1) * stride - 2 * padding + k.shape[0] + out_pad
            out_w = (arr.shape[1] - 1) * stride - 2 * padding + k.shape[1] + out_pad
            result = cp.zeros((out_h, out_w), dtype=arr.dtype)
            for i in range(arr.shape[0]):
                for j in range(arr.shape[1]):
                    region = result[
                        i * stride : i * stride + k.shape[0],
                        j * stride : j * stride + k.shape[1],
                    ]
                    region += arr[i, j] * k
            if use_torch:
                return torch.from_numpy(cp.asnumpy(result))
            if isinstance(value, np.ndarray) and not CUDA_AVAILABLE:
                return cp.asnumpy(result)
            return result
        elif self.neuron_type == "convtranspose1d":
            kernel = self.params.get("kernel", np.ones(1))
            stride = self.params.get("stride", 1)
            padding = self.params.get("padding", 0)
            out_pad = self.params.get("output_padding", 0)
            if torch.is_tensor(value):
                arr = value.detach().cpu().numpy()
                use_torch = True
            else:
                arr = cp.asarray(value)
                use_torch = False
            k = cp.asarray(kernel)
            out_len = (arr.shape[0] - 1) * stride - 2 * padding + k.shape[0] + out_pad
            result = cp.zeros((out_len,), dtype=arr.dtype)
            for i in range(arr.shape[0]):
                region = result[i * stride : i * stride + k.shape[0]]
                region += arr[i] * k
            if use_torch:
                return torch.from_numpy(cp.asnumpy(result))
            if isinstance(value, np.ndarray) and not CUDA_AVAILABLE:
                return cp.asnumpy(result)
            return result
        elif self.neuron_type == "conv3d":
            kernel = self.params.get("kernel", np.ones((1, 1, 1)))
            stride = self.params.get("stride", 1)
            padding = self.params.get("padding", 0)
            if torch.is_tensor(value):
                arr = value.detach().cpu().numpy()
                use_torch = True
            else:
                arr = cp.asarray(value)
                use_torch = False
            k = cp.asarray(kernel)
            if padding > 0:
                arr = cp.pad(
                    arr,
                    (
                        (padding, padding),
                        (padding, padding),
                        (padding, padding),
                    ),
                    mode="constant",
                )
            out_d = (arr.shape[0] - k.shape[0]) // stride + 1
            out_h = (arr.shape[1] - k.shape[1]) // stride + 1
            out_w = (arr.shape[2] - k.shape[2]) // stride + 1
            result = cp.zeros((out_d, out_h, out_w), dtype=arr.dtype)
            for i in range(out_d):
                for j in range(out_h):
                    for l in range(out_w):
                        region = arr[
                            i * stride : i * stride + k.shape[0],
                            j * stride : j * stride + k.shape[1],
                            l * stride : l * stride + k.shape[2],
                        ]
                        result[i, j, l] = cp.sum(region * k)
            if use_torch:
                return torch.from_numpy(cp.asnumpy(result))
            if isinstance(value, np.ndarray) and not CUDA_AVAILABLE:
                return cp.asnumpy(result)
            return result
        elif self.neuron_type == "convtranspose3d":
            kernel = self.params.get("kernel", np.ones((1, 1, 1)))
            stride = self.params.get("stride", 1)
            padding = self.params.get("padding", 0)
            out_pad = self.params.get("output_padding", 0)
            if torch.is_tensor(value):
                arr = value.detach().cpu().numpy()
                use_torch = True
            else:
                arr = cp.asarray(value)
                use_torch = False
            k = cp.asarray(kernel)
            out_d = (arr.shape[0] - 1) * stride - 2 * padding + k.shape[0] + out_pad
            out_h = (arr.shape[1] - 1) * stride - 2 * padding + k.shape[1] + out_pad
            out_w = (arr.shape[2] - 1) * stride - 2 * padding + k.shape[2] + out_pad
            result = cp.zeros((out_d, out_h, out_w), dtype=arr.dtype)
            for i in range(arr.shape[0]):
                for j in range(arr.shape[1]):
                    for l in range(arr.shape[2]):
                        region = result[
                            i * stride : i * stride + k.shape[0],
                            j * stride : j * stride + k.shape[1],
                            l * stride : l * stride + k.shape[2],
                        ]
                        region += arr[i, j, l] * k
            if use_torch:
                return torch.from_numpy(cp.asnumpy(result))
            if isinstance(value, np.ndarray) and not CUDA_AVAILABLE:
                return cp.asnumpy(result)
            return result
        elif self.neuron_type == "batchnorm":
            m = self.params.get("mean", 0.0)
            v = self.params.get("var", 1.0)
            mom = self.params.get("momentum", 0.1)
            eps = self.params.get("eps", 1e-5)
            self.params["mean"] = (1 - mom) * m + mom * value
            self.params["var"] = (1 - mom) * v + mom * (
                (value - self.params["mean"]) ** 2
            )
            return (value - self.params["mean"]) / ((self.params["var"] + eps) ** 0.5)
        elif self.neuron_type == "dropout":
            p = self.params.get("p", 0.5)
            return 0.0 if random.random() < p else value
        elif self.neuron_type == "dropout2d":
            p = self.params.get("p", 0.5)
            if torch.is_tensor(value):
                mask = (torch.rand_like(value, dtype=torch.float32) > p).float()
                return value * mask
            elif isinstance(value, cp.ndarray):
                mask = (cp.random.rand(*value.shape) > p).astype(value.dtype)
                return value * mask
            else:
                mask = (np.random.rand(*np.asarray(value).shape) > p).astype(float)
                return np.asarray(value) * mask
        elif self.neuron_type == "relu":
            if torch.is_tensor(value):
                return torch.relu(value)
            elif isinstance(value, cp.ndarray):
                return cp.maximum(value, 0)
            else:
                return max(0.0, value)
        elif self.neuron_type == "leakyrelu":
            slope = self.params.get("negative_slope", 0.01)
            if torch.is_tensor(value):
                return torch.where(value > 0, value, slope * value)
            elif isinstance(value, cp.ndarray):
                return cp.where(value > 0, value, slope * value)
            else:
                return value if value > 0 else slope * value
        elif self.neuron_type == "prelu":
            alpha = self.params.get("alpha", 0.25)
            if torch.is_tensor(value):
                return torch.where(value > 0, value, alpha * value)
            elif isinstance(value, cp.ndarray):
                return cp.where(value > 0, value, alpha * value)
            else:
                return value if value > 0 else alpha * value
        elif self.neuron_type == "elu":
            alpha = self.params.get("alpha", 1.0)
            if torch.is_tensor(value):
                return torch.where(value > 0, value, alpha * (torch.exp(value) - 1))
            elif isinstance(value, cp.ndarray):
                return cp.where(value > 0, value, alpha * (cp.exp(value) - 1))
            else:
                return value if value > 0 else alpha * (math.exp(value) - 1)
        elif self.neuron_type == "gelu":
            if torch.is_tensor(value):
                return torch.nn.functional.gelu(value)
            elif isinstance(value, cp.ndarray):
                return (
                    0.5
                    * value
                    * (
                        1.0
                        + cp.tanh(cp.sqrt(2.0 / cp.pi) * (value + 0.044715 * value**3))
                    )
                )
            else:
                return (
                    0.5
                    * value
                    * (
                        1.0
                        + math.tanh(
                            math.sqrt(2.0 / math.pi) * (value + 0.044715 * value**3)
                        )
                    )
                )
        elif self.neuron_type == "sigmoid":
            if torch.is_tensor(value):
                return torch.sigmoid(value)
            elif isinstance(value, cp.ndarray):
                return 1.0 / (1.0 + cp.exp(-value))
            else:
                return 1.0 / (1.0 + math.exp(-value))
        elif self.neuron_type == "tanh":
            if torch.is_tensor(value):
                return torch.tanh(value)
            elif isinstance(value, cp.ndarray):
                return cp.tanh(value)
            else:
                return math.tanh(value)
        elif self.neuron_type == "softmax":
            axis = self.params.get("axis", -1)
            if torch.is_tensor(value):
                v = value - torch.max(value)
                exp = torch.exp(v)
                return exp / torch.sum(exp, dim=axis, keepdim=True)
            elif isinstance(value, cp.ndarray):
                v = value - cp.max(value)
                exp = cp.exp(v)
                denom = cp.sum(exp, axis=axis, keepdims=True)
                return exp / denom
            else:
                v = value - np.max(value)
                exp = np.exp(v)
                return exp / np.sum(exp, axis=axis, keepdims=True)
        elif self.neuron_type in {"maxpool1d", "avgpool1d"}:
            size = self.params.get("size", 2)
            stride = self.params.get("stride", 2)
            self.value_history.append(value)
            if len(self.value_history) < size:
                return value
            window = self.value_history[-size:]
            if len(self.value_history) > size:
                self.value_history = self.value_history[stride:]
            if self.neuron_type == "maxpool1d":
                if torch.is_tensor(window[0]):
                    return torch.stack(window).max(dim=0).values
                elif isinstance(window[0], cp.ndarray):
                    stacked = cp.stack(window)
                    return cp.max(stacked, axis=0)
                else:
                    return max(window)
            else:
                if torch.is_tensor(window[0]):
                    return torch.stack(window).mean(dim=0)
                elif isinstance(window[0], cp.ndarray):
                    stacked = cp.stack(window)
                    return cp.mean(stacked, axis=0)
                else:
                    return sum(window) / len(window)
        elif self.neuron_type in {"maxpool2d", "avgpool2d"}:
            size = self.params.get("size", 2)
            stride = self.params.get("stride", 2)
            if torch.is_tensor(value):
                arr = value.detach().cpu().numpy()
                use_torch = True
            else:
                arr = cp.asarray(value)
                use_torch = False
            out_h = (arr.shape[0] - size) // stride + 1
            out_w = (arr.shape[1] - size) // stride + 1
            result = cp.zeros((out_h, out_w), dtype=arr.dtype)
            for i in range(out_h):
                for j in range(out_w):
                    region = arr[
                        i * stride : i * stride + size,
                        j * stride : j * stride + size,
                    ]
                    if self.neuron_type == "maxpool2d":
                        result[i, j] = cp.max(region)
                    else:
                        result[i, j] = cp.mean(region)
            if use_torch:
                return torch.from_numpy(cp.asnumpy(result))
            if isinstance(value, np.ndarray) and not CUDA_AVAILABLE:
                return cp.asnumpy(result)
            return result
        elif self.neuron_type == "flatten":
            if torch.is_tensor(value):
                return torch.reshape(value, (-1,))
            elif isinstance(value, cp.ndarray):
                return cp.reshape(value, (-1,))
            elif isinstance(value, np.ndarray):
                return value.reshape(-1)
            else:
                return value
        elif self.neuron_type == "embedding":
            weights = self.params.get("weights")
            if torch.is_tensor(value):
                idx = int(value.item()) % weights.shape[0]
                return torch.tensor(weights[idx], dtype=torch.float32)
            indices = np.asarray(value, dtype=int)
            idx = indices % weights.shape[0]
            result = weights[idx]
            if isinstance(value, cp.ndarray):
                return cp.asarray(result)
            return result
        elif self.neuron_type == "rnn":
            p = self.params
            h = self.hidden_state
            tanh = cp.tanh if isinstance(value, cp.ndarray) else math.tanh
            h = tanh(value * p["wx"] + h * p["wh"] + p["b"])
            self.hidden_state = h
            return h
        elif self.neuron_type == "lstm":
            p = self.params
            h = self.hidden_state
            c = self.cell_state
            sig = lambda x: (
                1.0 / (1.0 + cp.exp(-x))
                if isinstance(value, cp.ndarray)
                else 1.0 / (1.0 + math.exp(-x))
            )
            tanh = cp.tanh if isinstance(value, cp.ndarray) else math.tanh
            i = sig(value * p["wi"] + h * p["ui"] + p["bi"])
            f = sig(value * p["wf"] + h * p["uf"] + p["bf"])
            o = sig(value * p["wo"] + h * p["uo"] + p["bo"])
            g = tanh(value * p["wg"] + h * p["ug"] + p["bg"])
            c = f * c + i * g
            h = o * tanh(c)
            self.cell_state = c
            self.hidden_state = h
            return h
        elif self.neuron_type == "gru":
            p = self.params
            h = self.hidden_state
            sig = lambda x: (
                1.0 / (1.0 + cp.exp(-x))
                if isinstance(value, cp.ndarray)
                else 1.0 / (1.0 + math.exp(-x))
            )
            tanh = cp.tanh if isinstance(value, cp.ndarray) else math.tanh
            r = sig(value * p["wr"] + h * p["ur"] + p["br"])
            z = sig(value * p["wz"] + h * p["uz"] + p["bz"])
            n = tanh(value * p["wn"] + r * h * p["un"] + p["bn"])
            h = (1 - z) * n + z * h
            self.hidden_state = h
            return h
        elif self.neuron_type == "layernorm":
            axis = self.params.get("axis", -1)
            eps = self.params.get("eps", 1e-5)
            gain = self.params.get("gain", 1.0)
            bias = self.params.get("bias", 0.0)
            if torch.is_tensor(value):
                mean = torch.mean(value, dim=axis, keepdim=True)
                var = torch.var(value, dim=axis, keepdim=True, unbiased=False)
                return gain * (value - mean) / torch.sqrt(var + eps) + bias
            elif isinstance(value, cp.ndarray):
                mean = cp.mean(value, axis=axis, keepdims=True)
                var = cp.var(value, axis=axis, keepdims=True)
                return gain * (value - mean) / cp.sqrt(var + eps) + bias
            else:
                arr = np.asarray(value)
                mean = np.mean(arr, axis=axis, keepdims=True)
                var = np.var(arr, axis=axis, keepdims=True)
                result = gain * (arr - mean) / np.sqrt(var + eps) + bias
                return result
        return value


class Synapse:
    def __init__(
        self,
        source,
        target,
        weight=1.0,
        synapse_type="standard",
        fatigue: float = 0.0,
        frozen: bool = False,
        echo_length: int = 5,
        remote_core: "Core | None" = None,
        remote_target: int | None = None,
        phase: float = 0.0,
        dropout_prob: float = 0.5,
        momentum: float = 0.1,
    ):
        self.source = source
        self.target = target
        self.weight = weight
        self.potential = 1.0
        self.synapse_type = (
            synapse_type if synapse_type in SYNAPSE_TYPES else "standard"
        )
        self.created_at = datetime.now()
        self.fatigue = float(fatigue)
        self.frozen = bool(frozen)
        self.echo_buffer: deque[float] = deque(maxlen=int(max(1, echo_length)))
        self.remote_core = remote_core
        self.remote_target = remote_target
        self.phase = float(phase)
        self.visit_count = 0
        self.dropout_prob = float(dropout_prob)
        self.momentum = float(momentum)
        self.running_mean = 0.0
        self.running_var = 1.0

    def update_fatigue(self, increase: float, decay: float) -> None:
        """Update fatigue using a decay factor and additive increase."""
        self.fatigue = max(0.0, min(1.0, self.fatigue * decay + increase))

    def update_echo(self, value: float, decay: float) -> None:
        """Store a decayed copy of ``value`` in the echo buffer."""
        if isinstance(value, np.ndarray):
            val = float(np.mean(value))
        else:
            val = float(value)
        self.echo_buffer.append(val * decay)

    def get_echo_average(self) -> float:
        """Return the average of stored echo values."""
        if not self.echo_buffer:
            return 0.0
        return float(sum(self.echo_buffer) / len(self.echo_buffer))

    def effective_weight(self, context=None, global_phase: float = 0.0):
        """Return the weight modified according to ``synapse_type`` and context.

        ``global_phase`` allows phase-based modulation for experimental
        algorithms such as phase-gated Neuronenblitz."""
        if context is None:
            context = {}
        w = self.weight * math.cos(self.phase - global_phase)
        if self.synapse_type == "excitatory":
            w = abs(w)
        elif self.synapse_type == "inhibitory":
            w = -abs(w)
        elif self.synapse_type == "modulatory":
            mod = 1.0 + context.get("reward", 0.0) - context.get("stress", 0.0)
            w *= mod
        w *= max(0.0, 1.0 - self.fatigue)
        return w

    def apply_side_effects(self, core, source_value):
        """Apply any extra behaviour implied by ``synapse_type``."""
        if core is None:
            return
        if self.synapse_type == "mirror":
            core.neurons[self.source].value = source_value * self.weight
        elif self.synapse_type == "multi_neuron":
            extra_ids = random.sample(
                range(len(core.neurons)), k=min(2, len(core.neurons))
            )
            for idx in extra_ids:
                core.neurons[idx].value = source_value * self.weight
        elif self.synapse_type == "recurrent":
            val = core.neurons[self.source].value
            if val is None:
                val = 0.0
            core.neurons[self.source].value = val + source_value * self.weight
        elif self.synapse_type == "interconnection" and self.remote_core is not None:
            tgt_id = (
                self.remote_target if self.remote_target is not None else self.target
            )
            if 0 <= tgt_id < len(self.remote_core.neurons):
                self.remote_core.neurons[tgt_id].value = source_value * self.weight

    def transmit(self, source_value, core=None, context=None):
        """Compute the transmitted value and apply side effects."""
        self.apply_side_effects(core, source_value)
        w = self.effective_weight(context)
        if self.synapse_type == "dropout" and random.random() < self.dropout_prob:
            if torch.is_tensor(source_value):
                return source_value * 0
            return 0.0 if not isinstance(source_value, cp.ndarray) else source_value * 0
        out = (
            source_value * w if not torch.is_tensor(source_value) else source_value * w
        )
        if self.synapse_type == "batchnorm":
            if torch.is_tensor(out):
                arr = out.detach().cpu().numpy()
            elif isinstance(out, cp.ndarray):
                arr = cp.asnumpy(out)
            else:
                arr = np.asarray(out)
            mean = arr.mean()
            var = arr.var()
            self.running_mean = (
                self.momentum * mean + (1 - self.momentum) * self.running_mean
            )
            self.running_var = (
                self.momentum * var + (1 - self.momentum) * self.running_var
            )
            norm = (arr - self.running_mean) / np.sqrt(self.running_var + 1e-5)
            if torch.is_tensor(out):
                out = torch.as_tensor(norm, device=out.device)
            elif isinstance(out, cp.ndarray):
                out = cp.asarray(norm)
            else:
                out = float(norm)
        return out


from data_compressor import DataCompressor


class ShortTermMemory:
    """Ephemeral in-memory storage."""

    def __init__(self):
        self.data = {}

    def store(self, key: Hashable, value: Any) -> None:
        self.data[key] = value

    def retrieve(self, key: Hashable) -> Any:
        return self.data.get(key)

    def clear(self) -> None:
        self.data.clear()


class LongTermMemory:
    """Persistent storage backed by disk."""

    def __init__(self, path: str = "long_term_memory.pkl"):
        self.path = path
        self.data = {}
        if os.path.exists(self.path):
            with open(self.path, "rb") as f:
                try:
                    self.data = pickle.load(f)
                except Exception:
                    self.data = {}

    def store(self, key: Hashable, value: Any) -> None:
        self.data[key] = value
        with open(self.path, "wb") as f:
            pickle.dump(self.data, f)

    def retrieve(self, key: Hashable) -> Any:
        return self.data.get(key)


class MemorySystem:
    """Hierarchical memory with short- and long-term layers."""

    def __init__(
        self,
        long_term_path: str = "long_term_memory.pkl",
        threshold: float = 0.5,
        consolidation_interval: int = 10,
    ):
        self.short_term = ShortTermMemory()
        self.long_term = LongTermMemory(long_term_path)
        self.threshold = threshold
        self.consolidation_interval = consolidation_interval
        self._since_consolidation = 0

    def consolidate(self) -> None:
        for k, v in list(self.short_term.data.items()):
            self.long_term.store(k, v)
        self.short_term.clear()

    def choose_layer(self, context: dict) -> ShortTermMemory | LongTermMemory:
        if (
            context.get("arousal", 0) > self.threshold
            or context.get("reward", 0) > self.threshold
        ):
            return self.long_term
        return self.short_term

    def store(self, key: Hashable, value: Any, context: dict | None = None) -> None:
        layer = self.choose_layer(context or {})
        layer.store(key, value)
        self._since_consolidation += 1
        if self._since_consolidation >= self.consolidation_interval:
            self.consolidate()
            self._since_consolidation = 0

    def retrieve(self, key: Hashable) -> Any:
        val = self.short_term.retrieve(key)
        if val is None:
            val = self.long_term.retrieve(key)
        return val


class DataLoader:
    """Encode and decode arbitrary Python objects with optional plugins."""

    _encoders: dict[type, Callable[[Any], bytes]] = {}
    _decoders: dict[str, Callable[[bytes], Any]] = {}

    @classmethod
    def register_plugin(
        cls,
        typ: type,
        encoder: Callable[[Any], bytes],
        decoder: Callable[[bytes], Any],
    ) -> None:
        """Register custom encoder/decoder pair for ``typ``."""
        cls._encoders[typ] = encoder
        cls._decoders[typ.__name__] = decoder

    def __init__(
        self,
        compressor: DataCompressor | None = None,
        compression_level: int = 6,
        compression_enabled: bool = True,
        metrics_visualizer: "MetricsVisualizer | None" = None,
        tensor_dtype: str = "uint8",
        *,
        tokenizer: "Tokenizer | None" = None,
        track_metadata: bool = True,
        round_trip_penalty: float = 0.0,
        enable_round_trip_check: bool = False,
        quantization_bits: int = 0,
        sparse_threshold: float | None = None,
    ) -> None:
        self.compressor = (
            compressor
            if compressor is not None
            else DataCompressor(
                level=compression_level,
                compression_enabled=compression_enabled,
                quantization_bits=quantization_bits,
                sparse_threshold=sparse_threshold,
            )
        )
        self.metrics_visualizer = metrics_visualizer
        self.tensor_dtype = cp.dtype(tensor_dtype)
        self.tokenizer = tokenizer
        self.track_metadata = track_metadata
        self.round_trip_penalty = round_trip_penalty
        self.enable_round_trip_check = enable_round_trip_check

    def _objects_equal(self, a: Any, b: Any) -> bool:
        if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
            return np.array_equal(a, b)
        if isinstance(a, cp.ndarray) and isinstance(b, cp.ndarray):
            return bool(cp.all(a == b))
        if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
            return torch.equal(a, b)
        return a == b

    def encode(self, data: Any) -> np.ndarray:
        plugin_type = None
        for typ, enc in self._encoders.items():
            if isinstance(data, typ):
                plugin_type = typ.__name__
                data_bytes = enc(data)
                payload = {
                    "__marble_plugin__": {"type": plugin_type},
                    "payload": data_bytes,
                }
                serialized = pickle.dumps(payload)
                break
        else:
            tokenized = False
            original_type = data.__class__
            if self.tokenizer is not None and isinstance(data, str):
                ids = self.tokenizer.encode(data).ids
                data = np.asarray(ids, dtype=np.int32)
                tokenized = True
            if self.track_metadata:
                meta = {
                    "module": original_type.__module__,
                    "type": original_type.__name__,
                }
                if tokenized:
                    meta["tokenized"] = True
                payload = {"__marble_meta__": meta, "payload": data}
                serialized = pickle.dumps(payload)
            else:
                serialized = pickle.dumps(data)
        compressed = self.compressor.compress(serialized)
        if self.metrics_visualizer is not None:
            ratio = len(compressed) / max(len(serialized), 1)
            self.metrics_visualizer.update({"compression_ratio": ratio})
        array_module = cp if CUDA_AVAILABLE else np
        base = array_module.frombuffer(compressed, dtype=array_module.uint8)
        if (
            self.tensor_dtype != array_module.uint8
            and len(base) % self.tensor_dtype.itemsize == 0
        ):
            base = base.view(self.tensor_dtype)
        return base

    def decode(self, tensor: np.ndarray | "torch.Tensor") -> Any:
        compressed = tensor.tobytes()
        serialized = self.compressor.decompress(compressed)
        if self.metrics_visualizer is not None:
            ratio = len(compressed) / max(len(serialized), 1)
            self.metrics_visualizer.update({"compression_ratio": ratio})
        data = pickle.loads(serialized)
        if isinstance(data, dict):
            if "__marble_plugin__" in data and "payload" in data:
                plugin_meta = data["__marble_plugin__"]
                dtype = plugin_meta.get("type")
                decoder = self._decoders.get(dtype)
                if decoder is not None:
                    return decoder(data["payload"])
            if "__marble_meta__" in data and "payload" in data:
                meta = data["__marble_meta__"]
                payload = data["payload"]
                if meta.get("tokenized") and self.tokenizer is not None:
                    if isinstance(payload, np.ndarray):
                        tokens = payload.tolist()
                    else:
                        tokens = list(payload)
                    data = self.tokenizer.decode(tokens)
                else:
                    data = payload
        return data

    def round_trip_penalty_for(self, value: Any) -> float:
        if not self.enable_round_trip_check:
            return 0.0
        restored = self.decode(self.encode(value))
        if self._objects_equal(restored, value):
            return 0.0
        return self.round_trip_penalty

    def encode_array(self, array: np.ndarray) -> np.ndarray:
        """Encode a NumPy array (or PyTorch tensor) into a uint8 tensor using compression."""
        if isinstance(array, torch.Tensor):
            array = array.detach().cpu().numpy()
        compressed = self.compressor.compress_array(array)
        if self.metrics_visualizer is not None:
            ratio = len(compressed) / max(array.nbytes, 1)
            self.metrics_visualizer.update({"compression_ratio": ratio})
        array_module = cp if CUDA_AVAILABLE else np
        base = array_module.frombuffer(compressed, dtype=array_module.uint8)
        if (
            self.tensor_dtype != array_module.uint8
            and len(base) % self.tensor_dtype.itemsize == 0
        ):
            base = base.view(self.tensor_dtype)
        return base

    def decode_array(self, tensor: np.ndarray) -> np.ndarray:
        """Decode a tensor created by ``encode_array`` back to a NumPy array."""
        if isinstance(tensor, torch.Tensor):
            tensor = tensor.detach().cpu().numpy()
        elif isinstance(tensor, cp.ndarray):
            tensor = cp.asnumpy(tensor)
        if tensor.dtype != np.uint8:
            tensor = tensor.view(np.uint8)
        compressed = tensor.tobytes()
        array = self.compressor.decompress_array(compressed)
        if self.metrics_visualizer is not None:
            ratio = len(compressed) / max(array.nbytes, 1)
            self.metrics_visualizer.update({"compression_ratio": ratio})
        return array

    def encode_tensor(self, tensor: "torch.Tensor") -> "torch.Tensor":
        """Encode a PyTorch tensor using compression."""
        np_array = tensor.detach().cpu().numpy()
        encoded = self.encode_array(np_array)
        if isinstance(encoded, cp.ndarray):
            encoded = cp.asnumpy(encoded)
        return torch.from_numpy(encoded.copy())

    def decode_tensor(self, tensor: "torch.Tensor") -> "torch.Tensor":
        """Decode a tensor created by ``encode_tensor`` back to a PyTorch tensor."""
        np_tensor = tensor.detach().cpu().numpy()
        decoded = self.decode_array(np_tensor)
        if isinstance(decoded, cp.ndarray):
            decoded = cp.asnumpy(decoded)
        return torch.from_numpy(decoded.copy())


class Core:
    def __init__(
        self,
        params: dict | None = None,
        formula=None,
        formula_num_neurons=100,
        metrics_visualizer: "MetricsVisualizer | None" = None,
        width: int | None = None,
        height: int | None = None,
    ):
        print("Initializing MARBLE Core...")
        params = params.copy() if isinstance(params, dict) else {}
        if width is not None:
            params.setdefault("width", width)
        if height is not None:
            params.setdefault("height", height)
        params.setdefault("xmin", -2.0)
        params.setdefault("xmax", 1.0)
        params.setdefault("ymin", -1.5)
        params.setdefault("ymax", 1.5)
        params.setdefault("width", 30)
        params.setdefault("height", 30)
        params.setdefault("max_iter", 50)
        params.setdefault("representation_size", 4)
        params.setdefault("message_passing_alpha", 0.5)
        params.setdefault("vram_limit_mb", 0.5)
        params.setdefault("ram_limit_mb", 1.0)
        params.setdefault("disk_limit_mb", 10)
        seed = params.get("random_seed")
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            try:
                cp.random.seed(seed)
            except Exception:
                pass
        self.params = params
        self.metrics_visualizer = metrics_visualizer
        self.neuron_pool = MemoryPool(_neuron_factory)
        self.synapse_pool = MemoryPool(_synapse_factory)
        if "file" in TIER_REGISTRY:
            fpath = params.get("file_tier_path")
            if fpath is not None:
                TIER_REGISTRY["file"].file_path = fpath
                os.makedirs(os.path.dirname(fpath), exist_ok=True)
        rep_size = params.get("representation_size", _REP_SIZE)
        if rep_size <= 0:
            raise ValueError("representation_size must be positive")
        self.weight_init_strategy = params.get("weight_init_strategy", "uniform")
        configure_representation_size(
            rep_size,
            weight_strategy=self.weight_init_strategy,
            mean=params.get("weight_init_mean", 0.0),
            std=params.get("weight_init_std", 1.0),
            min_val=params.get("weight_init_min", 0.5),
            max_val=params.get("weight_init_max", 1.5),
        )
        self.rep_size = rep_size
        self.attention_module = AttentionModule(
            params.get("attention_temperature", 1.0)
        )
        self.weight_init_min = params.get("weight_init_min", 0.5)
        self.weight_init_max = params.get("weight_init_max", 1.5)
        self.weight_init_mean = params.get("weight_init_mean", 0.0)
        self.weight_init_std = params.get("weight_init_std", 1.0)
        self.weight_init_type = params.get("weight_init_type", "uniform")
        self.mandelbrot_escape_radius = params.get("mandelbrot_escape_radius", 2.0)
        self.mandelbrot_power = params.get("mandelbrot_power", 2)
        self.tier_autotune_enabled = params.get("tier_autotune_enabled", True)
        self.memory_cleanup_interval = params.get("memory_cleanup_interval", 60)
        self.representation_noise_std = params.get("representation_noise_std", 0.0)
        self.energy_threshold = params.get("energy_threshold", 0.0)
        self.rl_enabled = params.get("reinforcement_learning_enabled", False)
        self.rl_discount = params.get("rl_discount", 0.9)
        self.rl_learning_rate = params.get("rl_learning_rate", 0.1)
        self.rl_epsilon = params.get("rl_epsilon", 1.0)
        self.rl_epsilon_decay = params.get("rl_epsilon_decay", 0.95)
        self.rl_min_epsilon = params.get("rl_min_epsilon", 0.1)
        self.q_table = {}
        self.gradient_clip_value = params.get("gradient_clip_value", 1.0)
        self.synapse_weight_decay = params.get("synapse_weight_decay", 0.0)
        self.synapse_dropout_prob = params.get("synapse_dropout_prob", 0.0)
        self.synapse_batchnorm_momentum = params.get("synapse_batchnorm_momentum", 0.1)
        self.show_message_progress = params.get("show_message_progress", False)
        self.salience_weight = params.get("salience_weight", 1.0)
        mpi = params.get("message_passing_iterations", 1)
        if mpi <= 0:
            raise ValueError("message_passing_iterations must be positive")
        self.message_passing_iterations = mpi
        self.cluster_algorithm = params.get("cluster_algorithm", "kmeans")
        self.synapse_echo_length = params.get("synapse_echo_length", 5)
        self.synapse_echo_decay = params.get("synapse_echo_decay", 0.9)
        self.interconnection_prob = params.get("interconnection_prob", 0.05)
        self.global_phase_rate = params.get("global_phase_rate", 0.0)
        self.global_phase = 0.0
        self.vram_limit_mb = params.get("vram_limit_mb", 100)
        self.ram_limit_mb = params.get("ram_limit_mb", 500)
        self.disk_limit_mb = params.get("disk_limit_mb", 10000)
        self.neurons = []
        self.synapses = []
        nid = 0

        if formula is not None:
            try:
                expr = sp.sympify(formula, evaluate=False)
            except Exception as e:
                raise ValueError(f"Formula parsing failed: {e}")
            for i in range(formula_num_neurons):
                neuron = self.neuron_pool.allocate()
                neuron.__init__(nid, value=0.0, tier="vram", rep_size=self.rep_size)
                neuron.formula = expr
                self.neurons.append(neuron)
                nid += 1
        else:
            mandel_cpu = generate_seed(params)
            for val in mandel_cpu.flatten():
                neuron = self.neuron_pool.allocate()
                neuron.__init__(
                    nid,
                    value=float(val),
                    tier="vram",
                    rep_size=self.rep_size,
                )
                self.neurons.append(neuron)
                nid += 1

        num_neurons = len(self.neurons)
        for i in range(num_neurons - 1):
            weight = self._init_weight()
            self.add_synapse(
                self.neurons[i].id,
                self.neurons[i + 1].id,
                weight=weight,
                synapse_type="standard",
                echo_length=self.synapse_echo_length,
            )

        if not CUDA_AVAILABLE:
            for neuron in self.neurons:
                if neuron.tier == "vram":
                    neuron.tier = "ram"
            if "vram" in TIER_REGISTRY and "ram" in TIER_REGISTRY:
                TIER_REGISTRY["ram"].limit_mb += TIER_REGISTRY["vram"].limit_mb
                TIER_REGISTRY["vram"].limit_mb = 0
            self.ram_limit_mb += self.vram_limit_mb
            self.vram_limit_mb = 0
            print("CUDA not available: migrated VRAM tiers to RAM.")
        self.check_memory_usage()
        if self.tier_autotune_enabled:
            self.autotune_tiers()
        # Neuronenblitz instance attached to this core, if any
        self.neuronenblitz = None

    def attach_neuronenblitz(self, nb: "Neuronenblitz") -> None:
        """Attach a Neuronenblitz instance for bidirectional access.

        Parameters
        ----------
        nb:
            The :class:`~marble_neuronenblitz.Neuronenblitz` instance to attach.

        Notes
        -----
        Stores ``nb`` on ``self`` and ensures ``nb.core`` references this
        ``Core``. This enables both objects to look each other up without
        manual wiring by the caller.
        """

        # Detach any existing Neuronenblitz instance from this core
        if self.neuronenblitz is not None and self.neuronenblitz is not nb:
            if hasattr(self.neuronenblitz, "detach_core"):
                self.neuronenblitz.detach_core()
            else:
                old_nb = self.neuronenblitz
                self.neuronenblitz = None
                if getattr(old_nb, "core", None) is self:
                    old_nb.core = None

        # Ensure the incoming instance is detached from its previous core
        if getattr(nb, "core", None) not in (None, self):
            if hasattr(nb, "detach_core"):
                nb.detach_core()
            else:
                old_core = nb.core
                nb.core = None
                if getattr(old_core, "neuronenblitz", None) is nb:
                    old_core.neuronenblitz = None

        self.neuronenblitz = nb
        if getattr(nb, "core", None) is not self:
            nb.core = self

    def detach_neuronenblitz(self) -> None:
        """Detach any connected Neuronenblitz instance."""

        if self.neuronenblitz is not None:
            nb = self.neuronenblitz
            self.neuronenblitz = None
            if hasattr(nb, "detach_core"):
                nb.detach_core()
            elif getattr(nb, "core", None) is self:
                nb.core = None

    def _init_weight(self, fan_in: int = 1, fan_out: int = 1) -> float:
        """Return an initial synapse weight based on configuration."""
        if self.weight_init_type == "uniform":
            return random.uniform(self.weight_init_min, self.weight_init_max)
        if self.weight_init_type == "normal":
            return random.gauss(self.weight_init_mean, self.weight_init_std)
        if self.weight_init_type == "xavier_uniform":
            limit = math.sqrt(6.0 / (fan_in + fan_out))
            return random.uniform(-limit, limit)
        if self.weight_init_type == "xavier_normal":
            std = math.sqrt(2.0 / (fan_in + fan_out))
            return random.gauss(0.0, std)
        if self.weight_init_type == "kaiming_uniform":
            limit = math.sqrt(6.0 / fan_in)
            return random.uniform(-limit, limit)
        if self.weight_init_type == "kaiming_normal":
            std = math.sqrt(2.0 / fan_in)
            return random.gauss(0.0, std)
        if self.weight_init_type == "constant":
            return self.weight_init_mean
        raise ValueError(f"Unknown weight_init_type: {self.weight_init_type}")

    def get_average_age(self, items: list) -> float:
        now = datetime.now()
        if not items:
            return 0
        total_age = sum((now - item.created_at).total_seconds() for item in items)
        return total_age / len(items)

    def get_usage_by_tier(self, tier: str) -> float:
        neurons_in_tier = [n for n in self.neurons if n.tier == tier]
        synapses_in_tier = [
            s for s in self.synapses if self.neurons[s.source].tier == tier
        ]
        usage_bytes = len(neurons_in_tier) * 32 + len(synapses_in_tier) * 16
        return usage_bytes / (1024 * 1024)

    def get_memory_usage_metrics(self) -> dict[str, float]:
        """Return a dictionary with current memory usage metrics."""
        from system_metrics import get_gpu_memory_usage, get_system_memory_usage

        metrics = {
            "vram_usage": self.get_usage_by_tier("vram"),
            "ram_usage": self.get_usage_by_tier("ram"),
            "disk_usage": self.get_usage_by_tier("disk"),
            "system_memory": get_system_memory_usage(),
            "gpu_memory": get_gpu_memory_usage(),
        }
        return metrics

    def summary(self) -> dict[str, Any]:
        """Return high level statistics about the core."""

        metrics = self.get_memory_usage_metrics()
        tier_counts = {t: 0 for t in TIER_REGISTRY.keys()}
        for n in self.neurons:
            tier_counts[n.tier] = tier_counts.get(n.tier, 0) + 1
        return {
            "num_neurons": len(self.neurons),
            "num_synapses": len(self.synapses),
            "rep_size": self.rep_size,
            "memory": metrics,
            "tier_counts": tier_counts,
        }

    def check_memory_usage(self) -> None:
        metrics = self.get_memory_usage_metrics()
        print(
            "Memory usage - VRAM: {vram_usage:.2f} MB, RAM: {ram_usage:.2f} MB, Disk: {disk_usage:.2f} MB".format(
                **metrics
            )
        )
        if self.metrics_visualizer is not None:
            self.metrics_visualizer.update(metrics)

    def autotune_tiers(self) -> None:
        """Automatically migrate neurons between tiers when usage exceeds limits."""
        limits = {
            "vram": self.params.get(
                "vram_limit_mb", TIER_REGISTRY.get("vram", VramTier()).limit_mb
            ),
            "ram": self.params.get(
                "ram_limit_mb", TIER_REGISTRY.get("ram", RamTier()).limit_mb
            ),
        }

        def migrate(src: str, dst: str) -> None:
            limit = limits.get(src)
            if limit is None:
                return
            while self.get_usage_by_tier(src) > limit:
                candidates = [n for n in self.neurons if n.tier == src]
                if not candidates:
                    break
                oldest = min(candidates, key=lambda n: n.created_at)
                oldest.tier = dst

        migrate("vram", "ram")
        migrate("ram", "disk")
        self.check_memory_usage()

    def cleanup_unused_neurons(self) -> None:
        """Remove neurons without connections when early cleanup is enabled."""
        if not self.params.get("early_cleanup_enabled", False):
            return
        now = time.time()
        last = getattr(self, "_last_cleanup", 0.0)
        if now - last < self.memory_cleanup_interval:
            return
        self._last_cleanup = now
        to_remove = []
        for idx, neuron in enumerate(self.neurons):
            has_out = bool(neuron.synapses)
            has_in = any(s.target == neuron.id for s in self.synapses)
            if not has_out and not has_in and neuron.energy <= self.energy_threshold:
                to_remove.append(idx)
        if not to_remove:
            return
        remaining = []
        id_map = {}
        for idx, neuron in enumerate(self.neurons):
            if idx in to_remove:
                continue
            new_id = len(remaining)
            id_map[neuron.id] = new_id
            neuron.id = new_id
            remaining.append(neuron)
        new_syn = []
        for syn in self.synapses:
            if syn.source in to_remove or syn.target in to_remove:
                continue
            syn.source = id_map[syn.source]
            syn.target = id_map[syn.target]
            new_syn.append(syn)
        for neuron in remaining:
            neuron.synapses = [s for s in new_syn if s.source == neuron.id]
        for idx in to_remove:
            self.neuron_pool.release(self.neurons[idx])
        removed_syn = [
            s for s in self.synapses if s.source in to_remove or s.target in to_remove
        ]
        for syn in removed_syn:
            self.synapse_pool.release(syn)
        self.neurons = remaining
        self.synapses = new_syn

    def prune_unused_neurons(self) -> None:
        """Remove neurons that have no incoming or outgoing synapses."""
        used = {s.source for s in self.synapses} | {s.target for s in self.synapses}
        if len(used) == len(self.neurons):
            return
        remaining = []
        id_map = {}
        for idx, neuron in enumerate(self.neurons):
            if idx not in used:
                continue
            new_id = len(remaining)
            id_map[idx] = new_id
            neuron.id = new_id
            remaining.append(neuron)
        new_syn = []
        for syn in self.synapses:
            if syn.source not in id_map or syn.target not in id_map:
                continue
            syn.source = id_map[syn.source]
            syn.target = id_map[syn.target]
            new_syn.append(syn)
        for neuron in remaining:
            neuron.synapses = [s for s in new_syn if s.source == neuron.id]
        removed_ids = [i for i in range(len(self.neurons)) if i not in used]
        for idx in removed_ids:
            self.neuron_pool.release(self.neurons[idx])
        old_syn = self.synapses
        self.neurons = remaining
        self.synapses = new_syn
        for syn in old_syn:
            if syn not in new_syn:
                self.synapse_pool.release(syn)

    def add_synapse(
        self,
        source_id,
        target_id,
        weight=1.0,
        synapse_type="standard",
        frozen: bool = False,
        echo_length: int | None = None,
        phase: float = 0.0,
        dropout_prob: float | None = None,
        momentum: float | None = None,
    ):
        syn = self.synapse_pool.allocate()
        syn.__init__(
            source_id,
            target_id,
            weight=weight,
            synapse_type=synapse_type,
            frozen=frozen,
            echo_length=(
                self.synapse_echo_length if echo_length is None else echo_length
            ),
            phase=phase,
            dropout_prob=(
                self.synapse_dropout_prob if dropout_prob is None else dropout_prob
            ),
            momentum=(
                self.synapse_batchnorm_momentum if momentum is None else momentum
            ),
        )
        self.neurons[source_id].synapses.append(syn)
        self.synapses.append(syn)
        return syn

    def freeze_fraction_of_synapses(self, fraction: float) -> None:
        """Randomly mark a fraction of synapses as frozen."""
        if not 0.0 <= fraction <= 1.0:
            raise ValueError("fraction must be between 0 and 1")
        count = int(len(self.synapses) * fraction)
        to_freeze = random.sample(self.synapses, count)
        for syn in to_freeze:
            syn.frozen = True

    def apply_weight_decay(self) -> None:
        """Multiply all synapse weights by ``1 - synapse_weight_decay``."""
        decay = self.synapse_weight_decay
        if decay <= 0:
            return
        factor = 1.0 - decay
        for syn in self.synapses:
            syn.weight *= factor

    def get_detailed_status(self):
        status = {}
        for tier in TIER_REGISTRY.keys():
            neurons_in_tier = [
                n for n in self.neurons if n.tier.lower() == tier.lower()
            ]
            synapses_in_tier = [
                s
                for s in self.synapses
                if self.neurons[s.source].tier.lower() == tier.lower()
            ]
            avg_neuron_age = self.get_average_age(neurons_in_tier)
            avg_synapse_age = self.get_average_age(synapses_in_tier)
            status[tier] = {
                "neuron_count": len(neurons_in_tier),
                "synapse_count": len(synapses_in_tier),
                "memory_mb": self.get_usage_by_tier(tier),
                "avg_neuron_age_sec": avg_neuron_age,
                "avg_synapse_age_sec": avg_synapse_age,
            }
        return status

    def choose_new_tier(self):
        available_tiers = sorted(TIER_REGISTRY.values(), key=lambda t: t.order)

        # Honour an explicitly configured preferred tier first
        preferred = self.params.get("default_growth_tier")
        if preferred in TIER_REGISTRY:
            limit_key = f"{preferred.lower()}_limit_mb"
            limit = self.params.get(limit_key, TIER_REGISTRY[preferred].limit_mb)
            if limit is None or self.get_usage_by_tier(preferred) < limit:
                return preferred

        for tier in available_tiers:
            limit_key = f"{tier.name.lower()}_limit_mb"
            limit = self.params.get(limit_key, tier.limit_mb)
            if limit is None:
                continue
            usage = self.get_usage_by_tier(tier.name)
            if usage < limit:
                return tier.name
        return available_tiers[-1].name

    def expand(
        self,
        num_new_neurons=10,
        num_new_synapses=15,
        alternative_connection_prob=0.1,
        target_tier=None,
        neuron_types=None,
    ):
        if not isinstance(num_new_neurons, int) or not isinstance(
            num_new_synapses, int
        ):
            raise TypeError("num_new_neurons and num_new_synapses must be integers")
        if num_new_neurons < 0 or num_new_synapses < 0:
            raise ValueError(
                "num_new_neurons and num_new_synapses must be non-negative"
            )
        if not 0.0 <= alternative_connection_prob <= 1.0:
            raise ValueError("alternative_connection_prob must be between 0 and 1")
        if target_tier is not None and target_tier not in TIER_REGISTRY:
            raise ValueError(f"Unknown target_tier '{target_tier}'")
        if neuron_types is not None:
            if isinstance(neuron_types, list):
                if not neuron_types:
                    raise ValueError("neuron_types list cannot be empty")
                for n_type in neuron_types:
                    if n_type not in NEURON_TYPES:
                        raise ValueError(f"Unknown neuron type: {n_type}")
            else:
                if neuron_types not in NEURON_TYPES:
                    raise ValueError(f"Unknown neuron type: {neuron_types}")
        self.cleanup_unused_neurons()
        if target_tier is None:
            target_tier = self.choose_new_tier()
        start_id = len(self.neurons)
        for i in range(num_new_neurons):
            if isinstance(neuron_types, list):
                n_type = random.choice(neuron_types) if neuron_types else "standard"
            else:
                n_type = neuron_types if neuron_types is not None else "standard"
            neuron = self.neuron_pool.allocate()
            neuron.__init__(
                start_id + i,
                value=0.0,
                tier=target_tier,
                neuron_type=n_type,
                rep_size=self.rep_size,
            )
            self.neurons.append(neuron)
        for _ in range(num_new_synapses):
            src = random.choice(self.neurons).id
            tgt = random.choice(self.neurons).id
            if src != tgt:
                self.add_synapse(
                    src,
                    tgt,
                    weight=self._init_weight(),
                    synapse_type=random.choice(SYNAPSE_TYPES),
                    echo_length=self.synapse_echo_length,
                )

        print(
            f"Core expanded: {num_new_neurons} new neurons in tier '{target_tier}' and {num_new_synapses} new synapses added."
        )
        self.check_memory_usage()
        if self.tier_autotune_enabled:
            self.autotune_tiers()

    def increase_representation_size(self, delta: int = 1) -> None:
        """Increase representation dimensionality for all neurons."""
        if delta <= 0:
            return
        new_size = self.rep_size + delta
        configure_representation_size(
            new_size,
            weight_strategy=self.weight_init_strategy,
            mean=self.weight_init_mean,
            std=self.weight_init_std,
            min_val=self.weight_init_min,
            max_val=self.weight_init_max,
        )
        for neuron in self.neurons:
            neuron.representation = np.pad(neuron.representation, (0, delta))
        self.rep_size = new_size
        global_event_bus.publish("rep_size_changed", {"new_size": self.rep_size})

    def decrease_representation_size(self, delta: int = 1) -> None:
        """Decrease representation dimensionality for all neurons."""
        if delta <= 0:
            return
        if self.rep_size - delta < 1:
            delta = self.rep_size - 1
        if delta <= 0:
            return
        new_size = self.rep_size - delta
        configure_representation_size(
            new_size,
            weight_strategy=self.weight_init_strategy,
            mean=self.weight_init_mean,
            std=self.weight_init_std,
            min_val=self.weight_init_min,
            max_val=self.weight_init_max,
        )
        for neuron in self.neurons:
            neuron.representation = neuron.representation[:new_size]
        self.rep_size = new_size
        global_event_bus.publish("rep_size_changed", {"new_size": self.rep_size})

    # Built-in reinforcement learning utilities
    def enable_rl(self) -> None:
        """Enable Q-learning inside the core."""
        self.rl_enabled = True

    def disable_rl(self) -> None:
        """Disable Q-learning functionality."""
        self.rl_enabled = False

    def rl_select_action(self, state: Hashable, n_actions: int) -> int:
        """Return epsilon-greedy action from the Q-table."""
        if not self.rl_enabled:
            raise RuntimeError("reinforcement learning disabled")
        if random.random() < self.rl_epsilon:
            return random.randrange(n_actions)
        return int(
            np.argmax([self.q_table.get((state, a), 0.0) for a in range(n_actions)])
        )

    def rl_update(
        self,
        state: Hashable,
        action: int,
        reward: float,
        next_state: Hashable,
        done: bool,
        n_actions: int = 4,
    ) -> None:
        """Update Q-table using standard Q-learning."""
        if not self.rl_enabled:
            return
        current = self.q_table.get((state, action), 0.0)
        next_q = 0.0
        if not done:
            next_q = max(
                self.q_table.get((next_state, a), 0.0) for a in range(n_actions)
            )
        target = reward + self.rl_discount * next_q
        self.q_table[(state, action)] = current + self.rl_learning_rate * (
            target - current
        )
        self.rl_epsilon = max(
            self.rl_min_epsilon, self.rl_epsilon * self.rl_epsilon_decay
        )

    def reset_q_table(self) -> None:
        """Clear all stored Q-values."""

        self.q_table = {}

    def cluster_neurons(self, k=3):
        if not self.neurons:
            return
        min_k = int(self.params.get("min_cluster_k", 1))
        k = int(max(k, min_k))
        processed_vals = []
        for n in self.neurons:
            val = n.value
            if isinstance(val, np.ndarray):
                if val.size == 0:
                    val = 0.0
                else:
                    val = float(np.mean(val))
            elif val is None:
                val = float("nan")
            processed_vals.append(val)
        values = np.array(processed_vals, dtype=float)
        values[~np.isfinite(values)] = 0.0
        k = int(min(k, len(values)))
        if k < 1:
            return
        centers = [np.random.choice(values)]
        for _ in range(1, k):
            dists = np.array([min((v - c) ** 2 for c in centers) for v in values])
            total = dists.sum()
            if total <= 0 or not np.isfinite(total):
                centers.append(np.random.choice(values))
            else:
                probs = dists / total
                centers.append(np.random.choice(values, p=probs))
        centers = np.array(centers)
        for _ in range(5):
            assignments = [
                int(np.argmin([abs(v - c) for c in centers])) for v in values
            ]
            for i in range(k):
                cluster_vals = [values[j] for j, a in enumerate(assignments) if a == i]
                if cluster_vals:
                    centers[i] = sum(cluster_vals) / len(cluster_vals)
        for idx, neuron in enumerate(self.neurons):
            neuron.cluster_id = assignments[idx]

    def relocate_clusters(self, high=1.0, medium=0.1):
        cluster_scores = {}
        for neuron in self.neurons:
            cid = neuron.cluster_id
            if cid is None:
                continue
            cluster_scores.setdefault(cid, 0.0)
            cluster_scores[cid] += neuron.attention_score
        for cid, score in cluster_scores.items():
            if score > high:
                new_tier = "vram"
            elif score > medium:
                new_tier = "ram"
            else:
                new_tier = "disk"
            for neuron in self.neurons:
                if neuron.cluster_id == cid:
                    neuron.tier = new_tier

    def extract_subcore(self, neuron_ids):
        params = {
            "xmin": -2.0,
            "xmax": 1.0,
            "ymin": -1.5,
            "ymax": 1.5,
            "width": 1,
            "height": 1,
            "max_iter": 1,
            "vram_limit_mb": 0.1,
            "ram_limit_mb": 0.1,
            "disk_limit_mb": 0.1,
        }
        params["representation_size"] = self.rep_size
        subcore = Core(params, formula=None, formula_num_neurons=0)
        subcore.neurons = []
        subcore.synapses = []
        id_map = {}
        for i, nid in enumerate(neuron_ids):
            n = self.neurons[nid]
            new_n = Neuron(
                i,
                value=n.value,
                tier=n.tier,
                neuron_type=n.neuron_type,
                rep_size=self.rep_size,
            )
            new_n.representation = n.representation.copy()
            new_n.params = copy.deepcopy(n.params)
            new_n.value_history = list(n.value_history)
            subcore.neurons.append(new_n)
            id_map[nid] = i
        for syn in self.synapses:
            if syn.source in id_map and syn.target in id_map:
                ns = Synapse(
                    id_map[syn.source],
                    id_map[syn.target],
                    weight=syn.weight,
                    synapse_type=syn.synapse_type,
                )
                subcore.neurons[id_map[syn.source]].synapses.append(ns)
                subcore.synapses.append(ns)
        return subcore

    def check_finite_state(self) -> None:
        """Raise ``ValueError`` if any representation, value or weight is NaN/Inf."""
        for n in self.neurons:
            if not np.all(np.isfinite(n.representation)):
                raise ValueError("NaN or Inf encountered in neuron representation")
            if n.value is not None and not np.isfinite(n.value):
                raise ValueError("NaN or Inf encountered in neuron value")
        for s in self.synapses:
            if not np.isfinite(s.weight):
                raise ValueError("NaN or Inf encountered in synapse weight")

    def run_message_passing(
        self,
        metrics_visualizer: "MetricsVisualizer | None" = None,
        attention_module: "AttentionModule | None" = None,
        iterations: int | None = None,
    ) -> float:
        """Execute the message passing routine multiple times.

        Parameters
        ----------
        metrics_visualizer : MetricsVisualizer or None
            If provided, average change statistics are sent to this instance.
        attention_module : AttentionModule or None
            Optional custom attention module. If ``None`` the core's configured
            module is used.
        iterations : int, optional
            How often to call :func:`perform_message_passing`. When omitted the
            value from ``self.message_passing_iterations`` is used.

        Returns
        -------
        float
            Average representation change across all iterations.

        Notes
        -----
        The attribute ``global_phase`` is incremented by ``global_phase_rate``
        after each iteration to enable phase-based gating of synapses.
        """

        if iterations is None:
            iterations = self.message_passing_iterations
        if attention_module is None:
            attention_module = self.attention_module

        total_change = 0.0
        for _ in range(int(iterations)):
            total_change += perform_message_passing(
                self,
                metrics_visualizer=metrics_visualizer,
                attention_module=attention_module,
                global_phase=self.global_phase,
                show_progress=self.show_message_progress,
            )
            self.global_phase = (self.global_phase + self.global_phase_rate) % (
                2 * math.pi
            )

        self.cleanup_unused_neurons()
        self.apply_weight_decay()

        avg_change = total_change / max(int(iterations), 1)
        if metrics_visualizer is not None:
            metrics_visualizer.update({"avg_message_passing_change": avg_change})
        self.check_finite_state()
        return avg_change


def benchmark_message_passing(
    core: "Core", iterations: int = 100, warmup: int = 10
) -> tuple[int, float]:
    """Benchmark :func:`perform_message_passing` execution speed.

    Parameters
    ----------
    core : Core
        The :class:`Core` instance to benchmark.
    iterations : int, optional
        Number of timed iterations to run. ``100`` by default.
    warmup : int, optional
        Warm-up runs executed before timing starts to stabilise caches.

    Returns
    -------
    tuple[int, float]
        The number of iterations executed and the average seconds per
        iteration.
    """

    for _ in range(int(warmup)):
        perform_message_passing(core)

    start = time.perf_counter()
    for _ in range(int(iterations)):
        perform_message_passing(core)
    total = time.perf_counter() - start
    return int(iterations), total / max(int(iterations), 1)
