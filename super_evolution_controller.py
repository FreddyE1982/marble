class SuperEvolutionController:
    """Adjusts all configurable parameters using self-attention feedback."""

    def __init__(self, brain):
        self.brain = brain
        self.prev_loss = None
        self.prev_speed = None
        self.prev_complexity = None
        self.prev_resources = None
        self.history = []

    def record_metrics(self, loss, epoch_time):
        complexity = len(self.brain.core.neurons) + len(self.brain.core.synapses)
        resources = (
            self.brain.core.get_usage_by_tier("vram")
            + self.brain.core.get_usage_by_tier("ram")
            + self.brain.core.get_usage_by_tier("disk")
        )
        entry = {
            "loss": loss,
            "speed": epoch_time,
            "complexity": complexity,
            "resources": resources,
        }
        self.history.append(entry)
        if self.prev_loss is not None:
            self._adjust_parameters(loss, epoch_time, complexity, resources)
        self.prev_loss = loss
        self.prev_speed = epoch_time
        self.prev_complexity = complexity
        self.prev_resources = resources

    def _apply_factor(self, obj, attr, factor):
        val = getattr(obj, attr)
        if isinstance(val, (int, float)):
            setattr(obj, attr, val * factor)

    def _apply_factor_recursive(self, obj, factor, seen=None):
        if seen is None:
            seen = set()
        oid = id(obj)
        if oid in seen:
            return
        seen.add(oid)

        if isinstance(obj, dict):
            for k, v in obj.items():
                if isinstance(v, (int, float)):
                    obj[k] = v * factor
                else:
                    self._apply_factor_recursive(v, factor, seen)
            return

        if not hasattr(obj, "__dict__"):
            return

        for attr in vars(obj):
            if attr.startswith("_") or attr in (
                "core",
                "neuronenblitz",
                "dataloader",
                "lobe_manager",
                "super_evo_controller",
            ):
                continue
            try:
                val = getattr(obj, attr)
            except AttributeError:
                continue
            if isinstance(val, (int, float)):
                setattr(obj, attr, val * factor)
            else:
                self._apply_factor_recursive(val, factor, seen)

    def _adjust_parameters(self, loss, speed, complexity, resources):
        lobes = self.brain.lobe_manager.lobes
        if not lobes:
            return
        total = sum(l.attention_score for l in lobes) or 1.0
        attention = sum(l.attention_score for l in lobes) / total
        factor = 1.0
        if loss > self.prev_loss:
            factor = 1 + attention * 0.01
        elif speed > self.prev_speed:
            factor = 1 - attention * 0.01
        elif complexity > self.prev_complexity:
            factor = 1 - attention * 0.01
        elif resources > self.prev_resources:
            factor = 1 - attention * 0.01
        if factor == 1.0:
            return
        for key in list(self.brain.core.params.keys()):
            if isinstance(self.brain.core.params[key], (int, float)):
                self.brain.core.params[key] *= factor
                if hasattr(self.brain.core, key):
                    setattr(self.brain.core, key, self.brain.core.params[key])

        self._apply_factor_recursive(self.brain.neuronenblitz, factor)
        self._apply_factor_recursive(self.brain, factor)
