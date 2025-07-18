from marble_imports import *

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

class Neuron:
    def __init__(self, nid, value=0.0, tier='vram'):
        self.id = nid
        self.value = value
        self.tier = tier
        self.synapses = []
        self.formula = None
        self.created_at = datetime.now()

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


class DataLoader:
    def __init__(self, compressor: DataCompressor | None = None):
        self.compressor = compressor if compressor is not None else DataCompressor()

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

class Core:
    def __init__(self, params, formula=None, formula_num_neurons=100):
        print("Initializing MARBLE Core...")
        self.params = params
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
                neuron = Neuron(nid, value=0.0, tier='vram')
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
                self.neurons.append(Neuron(nid, value=float(val), tier='vram'))
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
              alternative_connection_prob=0.1, target_tier=None):
        if target_tier is None:
            target_tier = self.choose_new_tier()
        start_id = len(self.neurons)
        for i in range(num_new_neurons):
            self.neurons.append(Neuron(start_id + i, value=0.0, tier=target_tier))
        for _ in range(num_new_synapses):
            src = random.choice(self.neurons).id
            tgt = random.choice(self.neurons).id
            if src != tgt:
                syn = Synapse(src, tgt, weight=random.uniform(0.1, 1.0))
                self.neurons[src].synapses.append(syn)
                self.synapses.append(syn)
        print(f"Core expanded: {num_new_neurons} new neurons in tier '{target_tier}' and {num_new_synapses} new synapses added.")
        self.check_memory_usage()
