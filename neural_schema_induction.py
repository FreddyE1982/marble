from collections import Counter
from marble_imports import *
from marble_core import Core, Neuron
from marble_neuronenblitz import Neuronenblitz


class NeuralSchemaInductionLearner:
    """Discover and create schema neurons for frequently used paths."""

    def __init__(
        self,
        core: Core,
        nb: Neuronenblitz,
        support_threshold: int = 2,
        max_schema_size: int = 3,
    ) -> None:
        self.core = core
        self.nb = nb
        self.support_threshold = int(support_threshold)
        self.max_schema_size = int(max_schema_size)
        self.sequences: list[list[int]] = []
        self.schemas: dict[tuple[int, ...], int] = {}
        # disable weight learning
        self.nb.learning_rate = 0.0
        self.nb.weight_decay = 0.0

    def _record_sequence(self, path: list) -> None:
        if not path:
            return
        seq = [path[0].source]
        seq.extend(syn.target for syn in path)
        self.sequences.append(seq)

    def _frequent_patterns(self) -> list[tuple[int, ...]]:
        counts: Counter[tuple[int, ...]] = Counter()
        for seq in self.sequences:
            for length in range(2, self.max_schema_size + 1):
                for i in range(len(seq) - length + 1):
                    pat = tuple(seq[i : i + length])
                    counts[pat] += 1
        return [p for p, c in counts.items() if c >= self.support_threshold]

    def _create_schema(self, pattern: tuple[int, ...]) -> None:
        if pattern in self.schemas:
            return
        new_id = len(self.core.neurons)
        tier = self.core.choose_new_tier()
        neuron = Neuron(new_id, value=0.0, tier=tier, rep_size=self.core.rep_size)
        self.core.neurons.append(neuron)
        for nid in pattern:
            self.core.add_synapse(new_id, nid, weight=1.0)
            self.core.add_synapse(nid, new_id, weight=1.0)
        self.schemas[pattern] = new_id

    def _induce_schemas(self) -> None:
        for pattern in self._frequent_patterns():
            self._create_schema(pattern)

    def train_step(self, input_value: float) -> float:
        out, path = self.nb.dynamic_wander(float(input_value), apply_plasticity=False)
        self._record_sequence(path)
        self._induce_schemas()
        return float(out) if isinstance(out, (int, float)) else float(np.mean(out))

    def train(self, inputs: list[float], epochs: int = 1) -> None:
        for _ in range(int(epochs)):
            for inp in inputs:
                self.train_step(float(inp))
