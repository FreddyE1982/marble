from __future__ import annotations

import math
from typing import Any

import numpy as np

import tensor_backend as tb
from marble_base import MetricsVisualizer

try:  # optional dependency for progress bar
    from tqdm import tqdm  # type: ignore
except Exception:  # pragma: no cover - tqdm not installed
    tqdm = None


class AttentionModule:
    """Compute attention weights for message passing."""

    def __init__(
        self, temperature: float = 1.0, gating_cfg: dict | None = None
    ) -> None:
        self.temperature = temperature
        self.gating_cfg = gating_cfg or {}
        # persistent state for chaotic gating
        self._chaos_state = 0.5

    def _gating(self, n: int, xp) -> Any | None:
        """Return gating values for ``n`` keys or ``None`` if disabled."""

        if not self.gating_cfg.get("enabled") or n == 0:
            return None
        mode = self.gating_cfg.get("mode", "sine")
        if mode == "sine":
            freq = float(self.gating_cfg.get("frequency", 1.0))
            idx = xp.arange(n, dtype=xp.float32)
            return xp.sin(2 * xp.pi * freq * idx / max(n, 1))
        if mode == "chaos":
            r = float(self.gating_cfg.get("chaos", 3.7))
            r = max(0.0, min(r, 4.0))
            vals = []
            x = self._chaos_state
            for _ in range(n):
                x = r * x * (1 - x)
                vals.append(x)
            self._chaos_state = x
            return xp.array(vals, dtype=xp.float32)
        return None

    def compute(self, query: np.ndarray, keys: list[np.ndarray]) -> Any:
        xp = tb.xp()
        if not keys:
            return xp.array([])
        q = xp.nan_to_num(query, nan=0.0, posinf=0.0, neginf=0.0)
        ks = xp.stack([xp.nan_to_num(k, nan=0.0, posinf=0.0, neginf=0.0) for k in keys])
        dots = ks @ q / max(self.temperature, 1e-6)
        gate = self._gating(len(keys), xp)
        if gate is not None:
            dots = dots * gate
        exp = xp.exp(dots - xp.max(dots))
        return exp / xp.sum(exp)


def perform_message_passing(
    core,
    alpha: float | None = None,
    metrics_visualizer: "MetricsVisualizer | None" = None,
    attention_module: "AttentionModule | None" = None,
    *,
    global_phase: float | None = None,
    show_progress: bool = False,
) -> float:
    """Propagate representations across synapses using attention and backend ops."""
    xp = tb.xp()
    if alpha is None:
        alpha = core.params.get("message_passing_alpha", 0.5)
    if attention_module is None:
        temp = core.params.get("attention_temperature", 1.0)
        gating_cfg = core.params.get("attention_gating", {})
        attention_module = AttentionModule(temperature=temp, gating_cfg=gating_cfg)
    if global_phase is None:
        global_phase = getattr(core, "global_phase", 0.0)

    beta = core.params.get("message_passing_beta", 1.0)
    dropout = core.params.get("message_passing_dropout", 0.0)
    activation = core.params.get("representation_activation", "tanh")
    attn_dropout = core.params.get("attention_dropout", 0.0)
    energy_thr = core.params.get("energy_threshold", 0.0)
    noise_std = core.params.get("representation_noise_std", 0.0)
    causal = core.params.get("attention_causal", False)

    new_reps = [n.representation.copy() for n in core.neurons]
    old_reps = [n.representation.copy() for n in core.neurons]
    targets = core.neurons
    pbar = None
    if show_progress:
        if tqdm is None:
            raise RuntimeError("tqdm is required for progress display")
        pbar = tqdm(targets, desc="MsgPass", ncols=80)
        targets = pbar
    for target in targets:
        if target.energy < energy_thr:
            continue
        incoming = [
            s
            for s in core.synapses
            if s.target == target.id
            and core.neurons[s.source].energy >= energy_thr
            and (not causal or s.source <= target.id)
        ]
        if not incoming:
            continue
        neigh_reps = []
        for s in incoming:
            if dropout > 0 and float(tb.rand(())) < dropout:
                continue
            w = s.effective_weight(global_phase=global_phase)
            neigh_reps.append(core.neurons[s.source].representation * w)
            s.apply_side_effects(core, core.neurons[s.source].representation)
        if not neigh_reps:
            continue
        target_rep = target.representation
        attn = attention_module.compute(target_rep, neigh_reps)
        if attn.size == 0:
            continue
        if attn_dropout > 0:
            mask = tb.rand(attn.shape) >= attn_dropout
            if not bool(xp.any(mask)):
                continue
            attn = attn * mask
            sum_attn = attn.sum()
            if float(tb.to_numpy(sum_attn)) == 0.0:
                continue
            attn = attn / sum_attn
        agg = sum(attn[i] * neigh_reps[i] for i in range(len(neigh_reps)))
        ln_enabled = core.params.get("apply_layer_norm", True)
        mp_enabled = core.params.get("use_mixed_precision", False)
        from marble_core import _simple_mlp  # local import to avoid cycle

        interm = alpha * target.representation + (1 - alpha) * _simple_mlp(
            agg, activation, apply_layer_norm=ln_enabled, mixed_precision=mp_enabled
        )
        mag = float(tb.to_numpy(xp.linalg.norm(agg)))
        gate = 1.0 - math.exp(-mag)
        if alpha == 0:
            gate = 1.0 if mag > 0 else 0.0
        interm = gate * interm + (1 - gate) * target.representation
        updated = beta * interm + (1 - beta) * target.representation
        if noise_std > 0:
            updated = updated + tb.randn(updated.shape) * noise_std
        new_reps[target.id] = updated
    for idx, rep in enumerate(new_reps):
        core.neurons[idx].representation = rep
    diffs = [
        float(tb.to_numpy(xp.linalg.norm(new_reps[i] - old_reps[i])))
        for i in range(len(new_reps))
    ]
    avg_change = sum(diffs) / len(diffs) if diffs else 0.0
    if metrics_visualizer is not None:
        rep_matrix = xp.stack([n.representation for n in core.neurons])
        variance = float(tb.to_numpy(xp.var(rep_matrix)))
        metrics_visualizer.update(
            {
                "message_passing_change": avg_change,
                "representation_variance": variance,
            }
        )
    if pbar is not None:
        pbar.close()
    return avg_change
