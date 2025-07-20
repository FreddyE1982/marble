class DimensionalitySearch:
    """Monitor validation loss and neuron representations to expand dimensions."""

    def __init__(self, core, max_size=12, improvement_threshold=0.02,
                 plateau_epochs=2, metrics_visualizer=None):
        self.core = core
        self.max_size = max_size
        self.threshold = improvement_threshold
        self.plateau_epochs = plateau_epochs
        self.prev_loss = None
        self.no_improve = 0
        self.metrics_visualizer = metrics_visualizer

    def evaluate(self, loss: float) -> None:
        """Check loss and expand representation size if improvements stall."""
        if self.prev_loss is None:
            self.prev_loss = loss
            return
        if self.core.rep_size >= self.max_size:
            return
        rel_improve = (self.prev_loss - loss) / max(abs(self.prev_loss), 1e-8)
        reps = [n.representation for n in self.core.neurons]
        if reps:
            import numpy as np
            variance = float(np.var(np.vstack(reps)))
            if self.metrics_visualizer is not None:
                self.metrics_visualizer.update({"representation_variance": variance})
        if rel_improve < self.threshold:
            self.no_improve += 1
            if self.no_improve >= self.plateau_epochs:
                self.core.increase_representation_size(1)
                self.no_improve = 0
                self.prev_loss = loss
        else:
            self.no_improve = 0
            self.prev_loss = loss
