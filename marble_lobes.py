from marble_imports import *

class Lobe:
    def __init__(self, lid, neuron_ids=None):
        self.id = lid
        self.neuron_ids = neuron_ids if neuron_ids is not None else []
        self.attention_score = 0.0

class LobeManager:
    """Manages lobes and performs self-attention based optimizations."""

    def __init__(self, core):
        self.core = core
        self.lobes = []

    def genesis(self, neuron_ids):
        """Create a new lobe containing ``neuron_ids``."""
        lobe = Lobe(len(self.lobes), list(neuron_ids))
        self.lobes.append(lobe)
        return lobe

    def organize(self):
        """Organize neurons into lobes based on their cluster IDs."""
        clusters = {}
        for idx, neuron in enumerate(self.core.neurons):
            if neuron.cluster_id is not None:
                clusters.setdefault(neuron.cluster_id, []).append(idx)
        self.lobes = []
        for ids in clusters.values():
            self.genesis(ids)

    def update_attention(self):
        for lobe in self.lobes:
            lobe.attention_score = sum(self.core.neurons[n].attention_score for n in lobe.neuron_ids)

    def self_attention(self, loss):
        """Adjust neuron attention based on lobe-level attention and loss."""
        if not self.lobes:
            return
        self.update_attention()
        avg_att = sum(l.attention_score for l in self.lobes) / len(self.lobes)
        for lobe in self.lobes:
            if loss is None:
                continue
            if loss > 0 and lobe.attention_score < avg_att:
                factor = 1.05
            elif loss <= 0 and lobe.attention_score > avg_att:
                factor = 0.95
            else:
                factor = 1.0
            for nid in lobe.neuron_ids:
                self.core.neurons[nid].attention_score *= factor

    def select_high_attention(self, threshold=1.0):
        """Return neuron IDs belonging to lobes with attention above ``threshold``."""
        self.update_attention()
        selected = []
        for lobe in self.lobes:
            if lobe.attention_score > threshold:
                selected.extend(lobe.neuron_ids)
        return selected
