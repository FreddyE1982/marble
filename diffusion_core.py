from __future__ import annotations

from collections import deque
from typing import Any

import global_workspace
from neuromodulatory_system import NeuromodulatorySystem

from marble_imports import cp, random, math
from marble_imports import *  # noqa: F401,F403
from marble_core import Core, DataLoader, perform_message_passing
from marble_base import MetricsVisualizer
from marble_neuronenblitz import Neuronenblitz
from hybrid_memory import HybridMemory


class DiffusionCore(Core):
    """MARBLE core extension for diffusion-based generation."""

    def __init__(
        self,
        params: dict | None = None,
        *,
        diffusion_steps: int | None = None,
        noise_start: float | None = None,
        noise_end: float | None = None,
        noise_schedule: str | None = None,
        workspace_broadcast: bool | None = None,
        neuromodulatory_system: NeuromodulatorySystem | None = None,
        remote_client: Any | None = None,
        metrics_visualizer: "MetricsVisualizer | None" = None,
    ) -> None:
        params = params.copy() if params is not None else {}
        if diffusion_steps is not None:
            params.setdefault("diffusion_steps", diffusion_steps)
        if noise_start is not None:
            params.setdefault("noise_start", noise_start)
        if noise_end is not None:
            params.setdefault("noise_end", noise_end)
        if noise_schedule is not None:
            params.setdefault("noise_schedule", noise_schedule)
        if workspace_broadcast is not None:
            params.setdefault("workspace_broadcast", workspace_broadcast)
        super().__init__(params, metrics_visualizer=metrics_visualizer)
        self.diffusion_steps = int(self.params.get("diffusion_steps", 10))
        self.noise_start = float(self.params.get("noise_start", 1.0))
        self.noise_end = float(self.params.get("noise_end", 0.1))
        self.noise_schedule = self.params.get("noise_schedule", "linear")
        self.neuronenblitz = Neuronenblitz(self)
        self.loader = DataLoader()
        self.remote_client = remote_client
        self.neuromodulatory_system = neuromodulatory_system
        self.workspace_broadcast = bool(
            self.params.get("workspace_broadcast", False)
        )
        self.history: deque[float] = deque(maxlen=100)
        self.hybrid_memory: HybridMemory | None = None
        if "hybrid_memory" in self.params:
            hm = self.params["hybrid_memory"]
            self.hybrid_memory = HybridMemory(
                self,
                self.neuronenblitz,
                vector_path=hm.get("vector_store_path", "vector_store.pkl"),
                symbolic_path=hm.get("symbolic_store_path", "symbolic_memory.pkl"),
                max_entries=hm.get("max_entries", 1000),
            )

    def _noise_level(self, step: int) -> float:
        if self.noise_schedule == "linear":
            ratio = step / max(1, self.diffusion_steps - 1)
            return self.noise_start * (1 - ratio) + self.noise_end * ratio
        if self.noise_schedule == "cosine":
            ratio = (math.cos(math.pi * step / self.diffusion_steps) + 1) / 2
            return self.noise_end + (self.noise_start - self.noise_end) * ratio
        raise ValueError(f"Unknown noise_schedule: {self.noise_schedule}")

    def diffuse(self, data: Any) -> float:
        """Generate a value by iteratively denoising ``data``."""
        tensor = self.loader.encode(data)
        if hasattr(tensor, "mean"):
            value = float(cp.asnumpy(tensor).astype(float).mean())
        else:
            value = float(tensor)
        current = value
        for step in range(self.diffusion_steps):
            noise = random.gauss(0.0, self._noise_level(step))
            if self.neuromodulatory_system is not None:
                ctx = self.neuromodulatory_system.get_context()
                noise *= 1.0 + ctx.get("stress", 0.0) - ctx.get("reward", 0.0)
            out, path = self.neuronenblitz.dynamic_wander(current + noise)
            self.neuronenblitz.apply_weight_updates_and_attention(path, 0.0)
            perform_message_passing(self)
            if self.hybrid_memory is not None:
                self.hybrid_memory.store(f"step_{step}", out)
            if self.metrics_visualizer is not None:
                self.metrics_visualizer.update({"diffusion_noise": abs(noise)})
            current = out
        self.history.append(current)
        if (
            self.remote_client is not None
            and self.get_usage_by_tier("vram")
            > self.params.get("offload_threshold", 1.0)
        ):
            try:
                self.remote_client.offload(self)
            except Exception:
                pass
        if self.workspace_broadcast and global_workspace.workspace is not None:
            global_workspace.workspace.publish("diffusion_core", current)
        return current
