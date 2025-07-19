from marble_imports import *
from marble_base import MetricsVisualizer
import copy

# Representation size for GNN-style message passing
_REP_SIZE = 4


def _init_weights(rep_size: int):
    rs = np.random.RandomState(0)
    w1 = rs.randn(rep_size, 8) * 0.1
    b1 = rs.randn(8) * 0.1
    w2 = rs.randn(8, rep_size) * 0.1
    b2 = rs.randn(rep_size) * 0.1
    return w1, b1, w2, b2


_W1, _B1, _W2, _B2 = _init_weights(_REP_SIZE)


def configure_representation_size(rep_size: int) -> None:
    """Configure global representation size used for message passing."""
    global _REP_SIZE, _W1, _B1, _W2, _B2
    if rep_size != _REP_SIZE:
        _REP_SIZE = rep_size
        _W1, _B1, _W2, _B2 = _init_weights(rep_size)


def _apply_activation(arr: np.ndarray, activation: str) -> np.ndarray:
    """Return ``arr`` passed through the given activation function."""
    if activation == "relu":
        return np.maximum(arr, 0)
    if activation == "sigmoid":
        return 1.0 / (1.0 + np.exp(-arr))
    return np.tanh(arr)


def _simple_mlp(x: np.ndarray, activation: str = "tanh") -> np.ndarray:
    """Tiny MLP with one hidden layer and configurable activations."""
    # Handle potential NaNs or infinities to avoid runtime warnings
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    h = _apply_activation(x @ _W1 + _B1, activation)
    return _apply_activation(h @ _W2 + _B2, activation)


class AttentionModule:
    """Compute attention weights for message passing."""

    def __init__(self, temperature: float = 1.0) -> None:
        self.temperature = temperature

    def compute(self, query: np.ndarray, keys: list[np.ndarray]) -> np.ndarray:
        if not keys:
            return np.array([])
        q = np.nan_to_num(query, nan=0.0, posinf=0.0, neginf=0.0)
        ks = [np.nan_to_num(k, nan=0.0, posinf=0.0, neginf=0.0) for k in keys]
        dots = np.dot(ks, q) / max(self.temperature, 1e-6)
        shifted = np.clip(dots - np.max(dots), -50, 50)
        exps = np.exp(shifted)
        denom = exps.sum()
        if not np.isfinite(denom) or denom == 0:
            return np.ones(len(ks)) / len(ks)
        return exps / denom


def perform_message_passing(
    core,
    alpha: float | None = None,
    metrics_visualizer: "MetricsVisualizer | None" = None,
    attention_module: "AttentionModule | None" = None,
) -> float:
    """Propagate representations across synapses using attention.

    Parameters
    ----------
    core : Core
        The :class:`Core` instance containing neurons and synapses.
    alpha : float, optional
        Mixing factor between the current representation and the message-passing
        update. If ``None`` the value is read from ``core.params`` using the
        ``message_passing_alpha`` key (default ``0.5``).
    """

    if alpha is None:
        alpha = core.params.get("message_passing_alpha", 0.5)
    if attention_module is None:
        temp = core.params.get("attention_temperature", 1.0)
        attention_module = AttentionModule(temperature=temp)

    beta = core.params.get("message_passing_beta", 1.0)
    dropout = core.params.get("message_passing_dropout", 0.0)
    activation = core.params.get("representation_activation", "tanh")
    energy_thr = core.params.get("energy_threshold", 0.0)
    noise_std = core.params.get("representation_noise_std", 0.0)

    new_reps = [n.representation.copy() for n in core.neurons]
    old_reps = [n.representation.copy() for n in core.neurons]
    for target in core.neurons:
        if target.energy < energy_thr:
            continue
        incoming = [
            s
            for s in core.synapses
            if s.target == target.id and core.neurons[s.source].energy >= energy_thr
        ]
        if not incoming:
            continue
        neigh_reps = []
        for s in incoming:
            if dropout > 0 and random.random() < dropout:
                continue
            w = s.effective_weight()
            neigh_reps.append(core.neurons[s.source].representation * w)
            s.apply_side_effects(core, core.neurons[s.source].representation)
        if not neigh_reps:
            continue
        target_rep = target.representation
        attn = attention_module.compute(target_rep, neigh_reps)
        if attn.size == 0:
            continue
        agg = sum(attn[i] * neigh_reps[i] for i in range(len(neigh_reps)))
        interm = alpha * target.representation + (1 - alpha) * _simple_mlp(
            agg, activation
        )
        updated = beta * interm + (1 - beta) * target.representation
        if noise_std > 0:
            updated = updated + np.random.randn(*updated.shape) * noise_std
        new_reps[target.id] = updated
    for idx, rep in enumerate(new_reps):
        core.neurons[idx].representation = rep
    diffs = [
        float(np.linalg.norm(new_reps[i] - old_reps[i])) for i in range(len(new_reps))
    ]
    avg_change = float(np.mean(diffs)) if diffs else 0.0
    if metrics_visualizer is not None:
        metrics_visualizer.update({"message_passing_change": avg_change})
    return avg_change


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
]

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

    def process(self, data):
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

    def process(self, data):
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


class Neuron:
    def __init__(
        self, nid, value=0.0, tier="vram", neuron_type="standard", rep_size=_REP_SIZE
    ):
        self.id = nid
        self.value = value
        self.tier = tier
        self.neuron_type = neuron_type if neuron_type in NEURON_TYPES else "standard"
        self.synapses = []
        self.formula = None
        self.created_at = datetime.now()
        self.cluster_id = None
        self.attention_score = 0.0
        self.energy = 1.0
        self.representation = np.zeros(rep_size, dtype=float)
        self.params = {}
        self.value_history = []
        self.initialize_params()

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
            self.params = {"mean": 0.0, "var": 1.0, "momentum": 0.1, "eps": 1e-5}
        elif self.neuron_type == "dropout":
            self.params = {"p": 0.5}
        elif self.neuron_type == "leakyrelu":
            self.params = {"negative_slope": 0.01}
        elif self.neuron_type == "elu":
            self.params = {"alpha": 1.0}
        elif self.neuron_type == "softmax":
            self.params = {"axis": -1}
        elif self.neuron_type in {"relu", "sigmoid", "tanh", "flatten"}:
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
                    arr, ((padding, padding), (padding, padding)), mode="constant"
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
                    ((padding, padding), (padding, padding), (padding, padding)),
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
                        i * stride : i * stride + size, j * stride : j * stride + size
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
        self, source, target, weight=1.0, synapse_type="standard", fatigue=0.0
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

    def update_fatigue(self, increase: float, decay: float) -> None:
        """Update fatigue using a decay factor and additive increase."""
        self.fatigue = max(0.0, min(1.0, self.fatigue * decay + increase))

    def effective_weight(self, context=None):
        """Return the weight modified according to ``synapse_type`` and context."""
        if context is None:
            context = {}
        w = self.weight
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

    def transmit(self, source_value, core=None, context=None):
        """Compute the transmitted value and apply side effects."""
        self.apply_side_effects(core, source_value)
        w = self.effective_weight(context)
        if torch.is_tensor(source_value):
            return source_value * w
        elif isinstance(source_value, cp.ndarray):
            return source_value * w
        else:
            return source_value * w


def compute_mandelbrot(
    xmin,
    xmax,
    ymin,
    ymax,
    width,
    height,
    max_iter: int = 256,
    escape_radius: float = 2.0,
    power: int = 2,
):
    """Return a Mandelbrot set fragment as a 2D array.

    Parameters
    ----------
    xmin, xmax, ymin, ymax : float
        Bounds of the complex plane section.
    width, height : int
        Resolution of the output grid.
    max_iter : int, optional
        Maximum iteration count before declaring divergence.
    escape_radius : float, optional
        Absolute value beyond which points are marked as diverging.
    power : int, optional
        Exponent applied during iteration, allowing fractal variations.
    """

    x = cp.linspace(xmin, xmax, width)
    y = cp.linspace(ymin, ymax, height)
    X, Y = cp.meshgrid(x, y)
    C = X + 1j * Y
    Z = cp.zeros_like(C, dtype=cp.complex64)
    mandelbrot = cp.zeros(C.shape, dtype=cp.int32)
    for i in range(max_iter):
        mask = cp.abs(Z) <= escape_radius
        if not mask.any():
            break
        Z[mask] = Z[mask] ** power + C[mask]
        mandelbrot[mask] = i
    return mandelbrot


from data_compressor import DataCompressor


class ShortTermMemory:
    """Ephemeral in-memory storage."""

    def __init__(self):
        self.data = {}

    def store(self, key, value):
        self.data[key] = value

    def retrieve(self, key):
        return self.data.get(key)

    def clear(self):
        self.data.clear()


class LongTermMemory:
    """Persistent storage backed by disk."""

    def __init__(self, path="long_term_memory.pkl"):
        self.path = path
        self.data = {}
        if os.path.exists(self.path):
            with open(self.path, "rb") as f:
                try:
                    self.data = pickle.load(f)
                except Exception:
                    self.data = {}

    def store(self, key, value):
        self.data[key] = value
        with open(self.path, "wb") as f:
            pickle.dump(self.data, f)

    def retrieve(self, key):
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

    def consolidate(self):
        for k, v in list(self.short_term.data.items()):
            self.long_term.store(k, v)
        self.short_term.clear()

    def choose_layer(self, context):
        if (
            context.get("arousal", 0) > self.threshold
            or context.get("reward", 0) > self.threshold
        ):
            return self.long_term
        return self.short_term


class DataLoader:
    def __init__(
        self,
        compressor: DataCompressor | None = None,
        compression_level: int = 6,
        compression_enabled: bool = True,
        metrics_visualizer: "MetricsVisualizer | None" = None,
    ) -> None:
        self.compressor = (
            compressor
            if compressor is not None
            else DataCompressor(
                level=compression_level, compression_enabled=compression_enabled
            )
        )
        self.metrics_visualizer = metrics_visualizer

    def encode(self, data):
        serialized = pickle.dumps(data)
        compressed = self.compressor.compress(serialized)
        if self.metrics_visualizer is not None:
            ratio = len(compressed) / max(len(serialized), 1)
            self.metrics_visualizer.update({"compression_ratio": ratio})
        tensor = np.frombuffer(compressed, dtype=np.uint8)
        return tensor

    def decode(self, tensor):
        compressed = tensor.tobytes()
        serialized = self.compressor.decompress(compressed)
        if self.metrics_visualizer is not None:
            ratio = len(compressed) / max(len(serialized), 1)
            self.metrics_visualizer.update({"compression_ratio": ratio})
        data = pickle.loads(serialized)
        return data

    def encode_array(self, array: np.ndarray) -> np.ndarray:
        """Encode a NumPy array (or PyTorch tensor) into a uint8 tensor using compression."""
        if isinstance(array, torch.Tensor):
            array = array.detach().cpu().numpy()
        compressed = self.compressor.compress_array(array)
        if self.metrics_visualizer is not None:
            ratio = len(compressed) / max(array.nbytes, 1)
            self.metrics_visualizer.update({"compression_ratio": ratio})
        return np.frombuffer(compressed, dtype=np.uint8)

    def decode_array(self, tensor: np.ndarray) -> np.ndarray:
        """Decode a tensor created by ``encode_array`` back to a NumPy array."""
        if isinstance(tensor, torch.Tensor):
            tensor = tensor.detach().cpu().numpy()
        compressed = tensor.tobytes()
        array = self.compressor.decompress_array(compressed)
        if self.metrics_visualizer is not None:
            ratio = len(compressed) / max(array.nbytes, 1)
            self.metrics_visualizer.update({"compression_ratio": ratio})
        return array

    def encode_tensor(self, tensor: "torch.Tensor") -> "torch.Tensor":
        """Encode a PyTorch tensor using compression."""
        np_array = tensor.detach().cpu().numpy()
        encoded_np = self.encode_array(np_array)
        return torch.from_numpy(encoded_np.copy())

    def decode_tensor(self, tensor: "torch.Tensor") -> "torch.Tensor":
        """Decode a tensor created by ``encode_tensor`` back to a PyTorch tensor."""
        np_tensor = tensor.detach().cpu().numpy()
        decoded_np = self.decode_array(np_tensor)
        return torch.from_numpy(decoded_np.copy())


class Core:
    def __init__(self, params, formula=None, formula_num_neurons=100):
        print("Initializing MARBLE Core...")
        seed = params.get("random_seed")
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            try:
                cp.random.seed(seed)
            except Exception:
                pass
        self.params = params
        if "file" in TIER_REGISTRY:
            fpath = params.get("file_tier_path")
            if fpath is not None:
                TIER_REGISTRY["file"].file_path = fpath
                os.makedirs(os.path.dirname(fpath), exist_ok=True)
        rep_size = params.get("representation_size", _REP_SIZE)
        configure_representation_size(rep_size)
        self.rep_size = rep_size
        self.attention_module = AttentionModule(
            params.get("attention_temperature", 1.0)
        )
        self.weight_init_min = params.get("weight_init_min", 0.5)
        self.weight_init_max = params.get("weight_init_max", 1.5)
        self.mandelbrot_escape_radius = params.get("mandelbrot_escape_radius", 2.0)
        self.mandelbrot_power = params.get("mandelbrot_power", 2)
        self.tier_autotune_enabled = params.get("tier_autotune_enabled", True)
        self.memory_cleanup_interval = params.get("memory_cleanup_interval", 60)
        self.representation_noise_std = params.get("representation_noise_std", 0.0)
        self.energy_threshold = params.get("energy_threshold", 0.0)
        self.gradient_clip_value = params.get("gradient_clip_value", 1.0)
        self.message_passing_iterations = params.get("message_passing_iterations", 1)
        self.cluster_algorithm = params.get("cluster_algorithm", "kmeans")
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
                neuron = Neuron(nid, value=0.0, tier="vram", rep_size=self.rep_size)
                neuron.formula = expr
                self.neurons.append(neuron)
                nid += 1
        else:
            mandel_gpu = compute_mandelbrot(
                params["xmin"],
                params["xmax"],
                params["ymin"],
                params["ymax"],
                params["width"],
                params["height"],
                params.get("max_iter", 256),
                escape_radius=self.mandelbrot_escape_radius,
                power=self.mandelbrot_power,
            )
            mandel_cpu = cp.asnumpy(mandel_gpu)
            noise_std = params.get("init_noise_std", 0.0)
            if noise_std:
                mandel_cpu = mandel_cpu + np.random.randn(*mandel_cpu.shape) * noise_std
            for val in mandel_cpu.flatten():
                self.neurons.append(
                    Neuron(nid, value=float(val), tier="vram", rep_size=self.rep_size)
                )
                nid += 1

        num_neurons = len(self.neurons)
        for i in range(num_neurons - 1):
            weight = random.uniform(0.5, 1.5)
            self.add_synapse(
                self.neurons[i].id,
                self.neurons[i + 1].id,
                weight=weight,
                synapse_type="standard",
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

    def get_average_age(self, items):
        now = datetime.now()
        if not items:
            return 0
        total_age = sum((now - item.created_at).total_seconds() for item in items)
        return total_age / len(items)

    def get_usage_by_tier(self, tier):
        neurons_in_tier = [n for n in self.neurons if n.tier == tier]
        synapses_in_tier = [
            s for s in self.synapses if self.neurons[s.source].tier == tier
        ]
        usage_bytes = len(neurons_in_tier) * 32 + len(synapses_in_tier) * 16
        return usage_bytes / (1024 * 1024)

    def check_memory_usage(self):
        usage_vram = self.get_usage_by_tier("vram")
        usage_ram = self.get_usage_by_tier("ram")
        usage_disk = self.get_usage_by_tier("disk")
        print(
            f"Memory usage - VRAM: {usage_vram:.2f} MB, RAM: {usage_ram:.2f} MB, Disk: {usage_disk:.2f} MB"
        )

    def autotune_tiers(self) -> None:
        """Automatically migrate neurons between tiers when usage exceeds limits."""
        limits = {
            "vram": self.params.get("vram_limit_mb", TIER_REGISTRY.get("vram", VramTier()).limit_mb),
            "ram": self.params.get("ram_limit_mb", TIER_REGISTRY.get("ram", RamTier()).limit_mb),
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

    def add_synapse(self, source_id, target_id, weight=1.0, synapse_type="standard"):
        syn = Synapse(source_id, target_id, weight=weight, synapse_type=synapse_type)
        self.neurons[source_id].synapses.append(syn)
        self.synapses.append(syn)
        return syn

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
        if target_tier is None:
            target_tier = self.choose_new_tier()
        start_id = len(self.neurons)
        for i in range(num_new_neurons):
            if isinstance(neuron_types, list):
                n_type = random.choice(neuron_types) if neuron_types else "standard"
            else:
                n_type = neuron_types if neuron_types is not None else "standard"
            self.neurons.append(
                Neuron(
                    start_id + i,
                    value=0.0,
                    tier=target_tier,
                    neuron_type=n_type,
                    rep_size=self.rep_size,
                )
            )
        for _ in range(num_new_synapses):
            src = random.choice(self.neurons).id
            tgt = random.choice(self.neurons).id
            if src != tgt:
                self.add_synapse(
                    src,
                    tgt,
                    weight=random.uniform(0.1, 1.0),
                    synapse_type=random.choice(SYNAPSE_TYPES),
                )
        print(
            f"Core expanded: {num_new_neurons} new neurons in tier '{target_tier}' and {num_new_synapses} new synapses added."
        )
        self.check_memory_usage()
        if self.tier_autotune_enabled:
            self.autotune_tiers()

    def cluster_neurons(self, k=3):
        if not self.neurons:
            return
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
        k = int(min(k, len(values)))
        centers = np.random.choice(values, k, replace=False)
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
            )

        avg_change = total_change / max(int(iterations), 1)
        if metrics_visualizer is not None:
            metrics_visualizer.update({"avg_message_passing_change": avg_change})
        return avg_change
