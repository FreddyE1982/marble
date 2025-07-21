class NDimensionalTopologyManager:
    """Dynamically grow and shrink representation dimensions based on loss trends
    and a self-attention signal from Neuronenblitz."""

    def __init__(
        self,
        core,
        nb,
        enabled: bool = False,
        target_dimensions: int | None = None,
        attention_threshold: float = 0.5,
        loss_improve_threshold: float = 0.01,
        stagnation_epochs: int = 5,
        metrics_visualizer=None,
    ) -> None:
        self.core = core
        self.nb = nb
        self.enabled = bool(enabled)
        self.target_dimensions = (
            target_dimensions if target_dimensions is not None else core.rep_size
        )
        self.attention_threshold = float(attention_threshold)
        self.loss_improve_threshold = float(loss_improve_threshold)
        self.stagnation_epochs = int(stagnation_epochs)
        self.metrics_visualizer = metrics_visualizer
        self.loss_history: list[float] = []
        self.adding_dimension = False
        self.loss_at_add = None
        self.prev_size = core.rep_size

    def _self_attention_score(self) -> float:
        if not hasattr(self.nb, "type_attention"):
            return 0.0
        if not self.nb.type_attention:
            return 0.0
        return max(float(v) for v in self.nb.type_attention.values())

    def evaluate(self, loss: float) -> None:
        if not self.enabled:
            return
        self.loss_history.append(float(loss))
        if len(self.loss_history) > self.stagnation_epochs:
            self.loss_history.pop(0)

        if self.adding_dimension:
            if self.loss_at_add is None:
                self.loss_at_add = loss
                return
            rel_drop = (self.loss_at_add - loss) / max(abs(self.loss_at_add), 1e-8)
            if rel_drop >= self.loss_improve_threshold:
                self.adding_dimension = False
                self.loss_at_add = None
                self.prev_size = self.core.rep_size
                return
            if len(self.loss_history) >= self.stagnation_epochs and all(
                abs(self.loss_history[i] - self.loss_history[0]) < self.loss_improve_threshold
                for i in range(len(self.loss_history))
            ):
                self.core.decrease_representation_size(1)
                self.adding_dimension = False
                self.loss_at_add = None
                self.loss_history.clear()
            return

        if self.core.rep_size >= self.target_dimensions:
            return

        if len(self.loss_history) < self.stagnation_epochs:
            return

        if all(
            abs(self.loss_history[i] - self.loss_history[0]) < self.loss_improve_threshold
            for i in range(len(self.loss_history))
        ):
            attn = self._self_attention_score()
            if attn >= self.attention_threshold:
                self.core.increase_representation_size(1)
                self.adding_dimension = True
                self.loss_at_add = loss
                self.loss_history.clear()
                if self.metrics_visualizer is not None:
                    self.metrics_visualizer.update({"ndim": self.core.rep_size})
                return


