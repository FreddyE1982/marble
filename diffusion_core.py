from __future__ import annotations

from collections import deque
from typing import Any
import os

import global_workspace
from neuromodulatory_system import NeuromodulatorySystem

from marble_imports import cp, random, math
from marble_imports import *  # noqa: F401,F403
from marble_core import (
    Core,
    DataLoader,
    perform_message_passing,
    MemorySystem,
)
from marble_base import MetricsVisualizer
from marble_neuronenblitz import Neuronenblitz
from hybrid_memory import HybridMemory
from plugin_system import load_plugins
from neural_schema_induction import NeuralSchemaInductionLearner
from continuous_weight_field_learning import ContinuousWeightFieldLearner
from harmonic_resonance_learning import HarmonicResonanceLearner
from fractal_dimension_learning import FractalDimensionLearner
from activation_visualization import plot_activation_heatmap
import predictive_coding
import torch


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
        predictive_coding_params: dict | None = None,
        schema_induction_params: dict | None = None,
        memory_system: MemorySystem | None = None,
        cwfl_params: dict | None = None,
        harmonic_params: dict | None = None,
        fractal_params: dict | None = None,
        activation_output_dir: str | None = None,
        activation_colormap: str | None = None,
        plugin_dirs: list[str] | None = None,
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
        if activation_colormap is not None:
            params.setdefault("activation_colormap", activation_colormap)
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
        self.schema_learner: NeuralSchemaInductionLearner | None = None
        if schema_induction_params:
            self.schema_learner = NeuralSchemaInductionLearner(
                self,
                self.neuronenblitz,
                support_threshold=schema_induction_params.get("support_threshold", 2),
                max_schema_size=schema_induction_params.get("max_schema_size", 3),
            )
        self.predictive_coding = None
        if predictive_coding_params is not None:
            self.predictive_coding = predictive_coding.activate(
                self.neuronenblitz,
                num_layers=predictive_coding_params.get("num_layers", 2),
                latent_dim=predictive_coding_params.get("latent_dim", self.rep_size),
                learning_rate=predictive_coding_params.get("learning_rate", 0.001),
            )
        self.memory_system = memory_system
        self.cwfl: ContinuousWeightFieldLearner | None = None
        if cwfl_params is not None:
            self.cwfl = ContinuousWeightFieldLearner(
                self,
                self.neuronenblitz,
                num_basis=cwfl_params.get("num_basis", 10),
                bandwidth=cwfl_params.get("bandwidth", 1.0),
                reg_lambda=cwfl_params.get("reg_lambda", 0.01),
                learning_rate=cwfl_params.get("learning_rate", 0.01),
            )
        self.harmonic: HarmonicResonanceLearner | None = None
        if harmonic_params is not None:
            self.harmonic = HarmonicResonanceLearner(
                self,
                self.neuronenblitz,
                base_frequency=harmonic_params.get("base_frequency", 1.0),
                decay=harmonic_params.get("decay", 0.99),
            )
        self.fractal: FractalDimensionLearner | None = None
        if fractal_params is not None:
            self.fractal = FractalDimensionLearner(
                self,
                self.neuronenblitz,
                target_dimension=fractal_params.get("target_dimension", 4.0),
            )
        self.activation_output_dir = activation_output_dir
        self.activation_colormap = self.params.get("activation_colormap", "viridis")
        if plugin_dirs:
            load_plugins(plugin_dirs)

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
        if self.hybrid_memory is not None:
            retrieved = self.hybrid_memory.retrieve(current, top_k=1)
            if retrieved:
                current = float(retrieved[0][1])
        for step in range(self.diffusion_steps):
            noise = random.gauss(0.0, self._noise_level(step))
            if self.neuromodulatory_system is not None:
                ctx = self.neuromodulatory_system.get_context()
                noise *= 1.0 + ctx.get("stress", 0.0) - ctx.get("reward", 0.0)
            if self.predictive_coding is not None:
                _ = self.predictive_coding.step(
                    torch.tensor([current + noise], dtype=torch.float32)
                )
            if self.cwfl is not None:
                self.cwfl.train_step(current + noise, current)
            if self.harmonic is not None:
                self.harmonic.train_step(current + noise, current)
            if self.fractal is not None:
                self.fractal.train_step(current + noise, current)
            out, path = self.neuronenblitz.dynamic_wander(current + noise)
            if self.schema_learner is not None:
                self.schema_learner._record_sequence(path)
                self.schema_learner._induce_schemas()
            self.neuronenblitz.apply_weight_updates_and_attention(path, 0.0)
            perform_message_passing(self)
            if self.hybrid_memory is not None:
                self.hybrid_memory.store(f"step_{step}", out)
            if self.memory_system is not None:
                self.memory_system.store(
                    f"step_{step}",
                    out,
                    context=(self.neuromodulatory_system.get_context()
                             if self.neuromodulatory_system is not None
                             else {}),
                )
            if self.metrics_visualizer is not None:
                rep_matrix = cp.stack([n.representation for n in self.neurons])
                variance = float(cp.var(rep_matrix))
                self.metrics_visualizer.update(
                    {
                        "diffusion_noise": abs(noise),
                        "representation_variance": variance,
                    }
                )
            current = out
        self.history.append(current)
        if self.metrics_visualizer is not None:
            self.metrics_visualizer.update({"diffusion_output": current})
        if self.memory_system is not None:
            self.memory_system.store(
                "diffusion_output",
                current,
                context=(self.neuromodulatory_system.get_context()
                         if self.neuromodulatory_system is not None
                         else {}),
            )
            self.memory_system.consolidate()
        if self.activation_output_dir is not None:
            path = os.path.join(self.activation_output_dir, f"diffusion_{len(self.history)}.png")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            plot_activation_heatmap(self, path, cmap=self.activation_colormap)
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
