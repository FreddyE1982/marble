from marble_imports import *

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


def _simple_mlp(x: np.ndarray) -> np.ndarray:
    """Tiny MLP with one hidden layer and tanh activations."""
    h = np.tanh(x @ _W1 + _B1)
    return np.tanh(h @ _W2 + _B2)


def perform_message_passing(core, alpha: float | None = None) -> None:
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

    new_reps = [n.representation.copy() for n in core.neurons]
    for target in core.neurons:
        incoming = [s for s in core.synapses if s.target == target.id]
        if not incoming:
            continue
        neigh_reps = [core.neurons[s.source].representation * s.weight for s in incoming]
        if not neigh_reps:
            continue
        dots = np.array([float(np.dot(target.representation, nr)) for nr in neigh_reps])
        exps = np.exp(dots - np.max(dots))
        attn = exps / exps.sum()
        agg = sum(attn[i] * neigh_reps[i] for i in range(len(neigh_reps)))
        new_reps[target.id] = alpha * target.representation + (1 - alpha) * _simple_mlp(agg)
    for idx, rep in enumerate(new_reps):
        core.neurons[idx].representation = rep

# List of supported neuron types
NEURON_TYPES = ["standard", "excitatory", "inhibitory", "modulatory"]

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
    def __init__(self, nid, value=0.0, tier='vram', neuron_type='standard', rep_size=_REP_SIZE):
        self.id = nid
        self.value = value
        self.tier = tier
        self.neuron_type = neuron_type if neuron_type in NEURON_TYPES else 'standard'
        self.synapses = []
        self.formula = None
        self.created_at = datetime.now()
        self.cluster_id = None
        self.attention_score = 0.0
        self.representation = np.zeros(rep_size, dtype=float)

class Synapse:
    def __init__(self, source, target, weight=1.0):
        self.source = source
        self.target = target
        self.weight = weight
        self.potential = 1.0
        self.created_at = datetime.now()

def compute_mandelbrot(xmin, xmax, ymin, ymax, width, height, max_iter=256):
    x = cp.linspace(xmin, xmax, width)
    y = cp.linspace(ymin, ymax, height)
    X, Y = cp.meshgrid(x, y)
    C = X + 1j * Y
    Z = cp.zeros_like(C, dtype=cp.complex64)
    mandelbrot = cp.zeros(C.shape, dtype=cp.int32)
    for i in range(max_iter):
        mask = cp.abs(Z) <= 2
        Z[mask] = Z[mask] * Z[mask] + C[mask]
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

    def __init__(self, long_term_path="long_term_memory.pkl", threshold: float = 0.5):
        self.short_term = ShortTermMemory()
        self.long_term = LongTermMemory(long_term_path)
        self.threshold = threshold

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
    def __init__(self, compressor: DataCompressor | None = None, compression_level: int = 6):
        self.compressor = compressor if compressor is not None else DataCompressor(level=compression_level)

    def encode(self, data):
        serialized = pickle.dumps(data)
        compressed = self.compressor.compress(serialized)
        tensor = np.frombuffer(compressed, dtype=np.uint8)
        return tensor

    def decode(self, tensor):
        compressed = tensor.tobytes()
        serialized = self.compressor.decompress(compressed)
        data = pickle.loads(serialized)
        return data

    def encode_array(self, array: np.ndarray) -> np.ndarray:
        """Encode a NumPy array into a uint8 tensor using compression."""
        compressed = self.compressor.compress_array(array)
        return np.frombuffer(compressed, dtype=np.uint8)

    def decode_array(self, tensor: np.ndarray) -> np.ndarray:
        """Decode a tensor created by ``encode_array`` back to a NumPy array."""
        compressed = tensor.tobytes()
        return self.compressor.decompress_array(compressed)

class Core:
    def __init__(self, params, formula=None, formula_num_neurons=100):
        print("Initializing MARBLE Core...")
        self.params = params
        if 'file' in TIER_REGISTRY:
            fpath = params.get('file_tier_path')
            if fpath is not None:
                TIER_REGISTRY['file'].file_path = fpath
                os.makedirs(os.path.dirname(fpath), exist_ok=True)
        rep_size = params.get('representation_size', _REP_SIZE)
        configure_representation_size(rep_size)
        self.rep_size = rep_size
        self.vram_limit_mb = params.get('vram_limit_mb', 100)
        self.ram_limit_mb = params.get('ram_limit_mb', 500)
        self.disk_limit_mb = params.get('disk_limit_mb', 10000)
        self.neurons = []
        self.synapses = []
        nid = 0
        
        if formula is not None:
            try:
                expr = sp.sympify(formula, evaluate=False)
            except Exception as e:
                raise ValueError(f"Formula parsing failed: {e}")
            for i in range(formula_num_neurons):
                neuron = Neuron(nid, value=0.0, tier='vram', rep_size=self.rep_size)
                neuron.formula = expr
                self.neurons.append(neuron)
                nid += 1
        else:
            mandel_gpu = compute_mandelbrot(
                params['xmin'], params['xmax'],
                params['ymin'], params['ymax'],
                params['width'], params['height'],
                params.get('max_iter', 256)
            )
            mandel_cpu = cp.asnumpy(mandel_gpu)
            for val in mandel_cpu.flatten():
                self.neurons.append(Neuron(nid, value=float(val), tier='vram', rep_size=self.rep_size))
                nid += 1

        num_neurons = len(self.neurons)
        for i in range(num_neurons - 1):
            weight = random.uniform(0.5, 1.5)
            syn = Synapse(self.neurons[i].id, self.neurons[i+1].id, weight)
            self.neurons[i].synapses.append(syn)
            self.synapses.append(syn)

        if not CUDA_AVAILABLE:
            for neuron in self.neurons:
                if neuron.tier == 'vram':
                    neuron.tier = 'ram'
            if 'vram' in TIER_REGISTRY and 'ram' in TIER_REGISTRY:
                TIER_REGISTRY['ram'].limit_mb += TIER_REGISTRY['vram'].limit_mb
                TIER_REGISTRY['vram'].limit_mb = 0
            self.ram_limit_mb += self.vram_limit_mb
            self.vram_limit_mb = 0
            print("CUDA not available: migrated VRAM tiers to RAM.")
        self.check_memory_usage()

    def get_average_age(self, items):
        now = datetime.now()
        if not items:
            return 0
        total_age = sum((now - item.created_at).total_seconds() for item in items)
        return total_age / len(items)

    def get_usage_by_tier(self, tier):
        neurons_in_tier = [n for n in self.neurons if n.tier == tier]
        synapses_in_tier = [s for s in self.synapses if self.neurons[s.source].tier == tier]
        usage_bytes = len(neurons_in_tier) * 32 + len(synapses_in_tier) * 16
        return usage_bytes / (1024 * 1024)

    def check_memory_usage(self):
        usage_vram = self.get_usage_by_tier('vram')
        usage_ram  = self.get_usage_by_tier('ram')
        usage_disk = self.get_usage_by_tier('disk')
        print(f"Memory usage - VRAM: {usage_vram:.2f} MB, RAM: {usage_ram:.2f} MB, Disk: {usage_disk:.2f} MB")

    def get_detailed_status(self):
        status = {}
        for tier in TIER_REGISTRY.keys():
            neurons_in_tier = [n for n in self.neurons if n.tier.lower() == tier.lower()]
            synapses_in_tier = [s for s in self.synapses if self.neurons[s.source].tier.lower() == tier.lower()]
            avg_neuron_age = self.get_average_age(neurons_in_tier)
            avg_synapse_age = self.get_average_age(synapses_in_tier)
            status[tier] = {
                'neuron_count': len(neurons_in_tier),
                'synapse_count': len(synapses_in_tier),
                'memory_mb': self.get_usage_by_tier(tier),
                'avg_neuron_age_sec': avg_neuron_age,
                'avg_synapse_age_sec': avg_synapse_age
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

    def expand(self, num_new_neurons=10, num_new_synapses=15,
              alternative_connection_prob=0.1, target_tier=None,
              neuron_types=None):
        if target_tier is None:
            target_tier = self.choose_new_tier()
        start_id = len(self.neurons)
        for i in range(num_new_neurons):
            if isinstance(neuron_types, list):
                n_type = random.choice(neuron_types) if neuron_types else 'standard'
            else:
                n_type = neuron_types if neuron_types is not None else 'standard'
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
                syn = Synapse(src, tgt, weight=random.uniform(0.1, 1.0))
                self.neurons[src].synapses.append(syn)
                self.synapses.append(syn)
        print(f"Core expanded: {num_new_neurons} new neurons in tier '{target_tier}' and {num_new_synapses} new synapses added.")
        self.check_memory_usage()

    def cluster_neurons(self, k=3):
        if not self.neurons:
            return
        values = np.array([n.value for n in self.neurons], dtype=float)
        k = min(k, len(values))
        centers = np.random.choice(values, k, replace=False)
        for _ in range(5):
            assignments = [int(np.argmin([abs(v - c) for c in centers])) for v in values]
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
                new_tier = 'vram'
            elif score > medium:
                new_tier = 'ram'
            else:
                new_tier = 'disk'
            for neuron in self.neurons:
                if neuron.cluster_id == cid:
                    neuron.tier = new_tier
                    neuron.attention_score = 0.0

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
            subcore.neurons.append(new_n)
            id_map[nid] = i
        for syn in self.synapses:
            if syn.source in id_map and syn.target in id_map:
                ns = Synapse(id_map[syn.source], id_map[syn.target], weight=syn.weight)
                subcore.neurons[id_map[syn.source]].synapses.append(ns)
                subcore.synapses.append(ns)
        return subcore

