class SuperEvolutionController:
    """Adjusts all configurable parameters using self-attention feedback.

    Parameters listed in ``BLOCKED_NAMES`` or matching any prefix in
    ``BLOCKED_PREFIXES`` are left untouched. Any attribute containing
    ``"val_loss"`` is also ignored.
    """

    BLOCKED_NAMES = {
        "auto_save_interval",
        "benchmark_interval",
        "loss_growth_threshold",
        "metrics_history_size",
        "last_val_loss",
        "best_validation_loss",
        "gradient_clip_value",
        "weight_init_min",
        "weight_init_max",
        "random_seed",
        "loss_scale",
    }

    BLOCKED_PREFIXES = ("tier_decision_params",)

    def __init__(self, brain):
        self.brain = brain
        self.prev_loss = None
        self.prev_speed = None
        self.prev_complexity = None
        self.prev_resources = None
        self.history = []
        self.change_log = []

    def _record_change(self, name, old, new):
        if old != new:
            self.change_log.append({"parameter": name, "old": old, "new": new})

    def _should_skip(self, name: str, prefix: str) -> bool:
        full = f"{prefix}{name}" if prefix else name
        if any(full.startswith(p) for p in self.BLOCKED_PREFIXES):
            return True
        if name in self.BLOCKED_NAMES:
            return True
        if "val_loss" in full:
            return True
        return False

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
        if isinstance(val, (int, float)) and not isinstance(val, bool):
            setattr(obj, attr, val * factor)

    def _apply_factor_recursive(self, obj, factor, seen=None, prefix=""):
        if seen is None:
            seen = set()
        oid = id(obj)
        if oid in seen:
            return
        seen.add(oid)

        if isinstance(obj, dict):
            for k, v in obj.items():
                if self._should_skip(k, prefix):
                    continue
                if isinstance(v, (int, float)):
                    old = v
                    obj[k] = v * factor
                    self._record_change(f"{prefix}{k}", old, obj[k])
                else:
                    self._apply_factor_recursive(
                        v, factor, seen, prefix=f"{prefix}{k}."
                    )
            return

        if not hasattr(obj, "__dict__"):
            return

        for attr in vars(obj):
            if attr.startswith("_") or attr in (
                "core",
                "neuronenblitz",
                "super_evo_controller",
            ):
                continue
            if self._should_skip(attr, prefix):
                continue
            try:
                val = getattr(obj, attr)
            except AttributeError:
                continue
            if isinstance(val, (int, float)) and not isinstance(val, bool):
                old = val
                new_val = val * factor
                setattr(obj, attr, new_val)
                self._record_change(f"{prefix}{attr}", old, new_val)
            else:
                self._apply_factor_recursive(
                    val, factor, seen, prefix=f"{prefix}{attr}."
                )

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
            if self._should_skip(key, "core.params."):
                continue
            val = self.brain.core.params[key]
            if isinstance(val, (int, float)) and not isinstance(val, bool):
                new_val = val * factor
                self.brain.core.params[key] = new_val
                self._record_change(f"core.params.{key}", val, new_val)
                if hasattr(self.brain.core, key):
                    setattr(self.brain.core, key, new_val)

        self._apply_factor_recursive(self.brain.neuronenblitz, factor)
        self._apply_factor_recursive(self.brain, factor)
