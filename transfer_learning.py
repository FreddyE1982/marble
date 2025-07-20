from marble_core import Core, perform_message_passing
from marble_neuronenblitz import Neuronenblitz

class TransferLearner:
    """Fine-tune a pretrained network while freezing a subset of synapses."""

    def __init__(self, core: Core, nb: Neuronenblitz, freeze_fraction: float = 0.5) -> None:
        if not 0.0 <= freeze_fraction <= 1.0:
            raise ValueError("freeze_fraction must be between 0 and 1")
        self.core = core
        self.nb = nb
        self.freeze_fraction = float(freeze_fraction)
        self.core.freeze_fraction_of_synapses(self.freeze_fraction)
        self.history: list[dict] = []

    def train_step(self, input_value: float, target_value: float) -> float:
        output, path = self.nb.dynamic_wander(input_value)
        error = target_value - output
        self.nb.apply_weight_updates_and_attention(path, error)
        perform_message_passing(self.core)
        loss = float(error * error)
        self.history.append({"loss": loss})
        return loss

    def train(self, examples: list[tuple[float, float]], epochs: int = 1) -> None:
        for _ in range(int(epochs)):
            for inp, tgt in examples:
                self.train_step(float(inp), float(tgt))
