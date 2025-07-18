class MetaParameterController:
    """Dynamically adjusts Neuronenblitz parameters based on performance history."""

    def __init__(self, history_length=5, adjustment=0.5,
                 min_threshold=1.0, max_threshold=20.0):
        self.history_length = history_length
        self.adjustment = adjustment
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        self.loss_history = []

    def record_loss(self, loss: float) -> None:
        """Record a new validation loss value."""
        self.loss_history.append(loss)
        if len(self.loss_history) > self.history_length:
            self.loss_history.pop(0)

    def adjust(self, neuronenblitz) -> None:
        """Adjust neuronenblitz.plasticity_threshold based on recent losses."""
        if len(self.loss_history) < 2:
            return
        last = self.loss_history[-1]
        prev = self.loss_history[-2]
        if last > prev:
            new_val = max(self.min_threshold,
                           neuronenblitz.plasticity_threshold - self.adjustment)
        else:
            new_val = min(self.max_threshold,
                           neuronenblitz.plasticity_threshold + self.adjustment)
        neuronenblitz.plasticity_threshold = new_val
