"""Evolutionary hyperparameter optimization utilities."""

from __future__ import annotations

import copy
import json
import math
import os
import random
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional

import networkx as nx
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    import torch
except Exception:  # pragma: no cover - torch may be optional
    torch = None


@dataclass
class Candidate:
    """Represents a configuration under evaluation."""

    id: int
    config: Dict[str, Any]
    parent_id: Optional[int] = None
    fitness: Optional[float] = None


class EvolutionTrainer:
    """Explore configuration space using evolutionary strategies.

    The trainer maintains a population of candidate configurations. Each
    generation mutates candidates, evaluates their fitness in parallel and
    selects the strongest individuals to seed the next generation.
    """

    def __init__(
        self,
        base_config: Dict[str, Any],
        train_func: Callable[[Dict[str, Any], int, str], float],
        mutation_space: Dict[str, Dict[str, Any]],
        population_size: int,
        selection_size: int,
        generations: int,
        *,
        mutation_rate: float = 0.1,
        steps_per_candidate: int = 5,
        parallelism: Optional[int] = None,
    ) -> None:
        if population_size < 1 or selection_size < 1 or generations < 1:
            raise ValueError("population_size, selection_size and generations must be positive")
        if not 0.0 <= mutation_rate <= 1.0:
            raise ValueError("mutation_rate must be in [0,1]")
        if selection_size > population_size:
            raise ValueError("selection_size cannot exceed population_size")
        self.base_config = copy.deepcopy(base_config)
        self.train_func = train_func
        self.mutation_space = mutation_space
        self.population_size = population_size
        self.selection_size = selection_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.steps_per_candidate = steps_per_candidate
        self.parallelism = parallelism or os.cpu_count() or 1
        self.device = "cuda" if torch is not None and torch.cuda.is_available() else "cpu"
        self._candidate_counter = 0
        self.graph = nx.DiGraph()
        self.lineage: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Mutation operators
    # ------------------------------------------------------------------
    def mutate(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Return a mutated copy of ``config`` according to ``mutation_space``."""
        mutated = copy.deepcopy(config)
        for key, spec in self.mutation_space.items():
            if random.random() > self.mutation_rate:
                continue
            if spec.get("type") in {"float", "int"}:
                base = float(mutated.get(key, self.base_config.get(key, 0.0)))
                sigma = float(spec.get("sigma", 0.1))
                new_val = base * math.exp(random.gauss(0.0, sigma))
                if spec["type"] == "int":
                    new_val = int(round(new_val))
                minv = spec.get("min")
                maxv = spec.get("max")
                if minv is not None:
                    new_val = max(minv, new_val)
                if maxv is not None:
                    new_val = min(maxv, new_val)
                mutated[key] = new_val
            elif spec.get("type") == "categorical":
                options = list(spec.get("options", []))
                if not options:
                    continue
                current = mutated.get(key, self.base_config.get(key))
                choices = [o for o in options if o != current] or options
                mutated[key] = random.choice(choices)
            else:
                raise ValueError(f"Unknown mutation type for {key}: {spec}")
        return mutated

    # ------------------------------------------------------------------
    # Evaluation hooks
    # ------------------------------------------------------------------
    def evaluate(self, candidate: Candidate) -> float:
        """Train ``candidate.config`` for a few steps and return fitness."""
        fitness = float(self.train_func(candidate.config, self.steps_per_candidate, self.device))
        candidate.fitness = fitness
        self.graph.add_node(candidate.id, fitness=fitness)
        if candidate.parent_id is not None:
            self.graph.add_edge(candidate.parent_id, candidate.id)
        self.lineage.append({
            "id": candidate.id,
            "parent": candidate.parent_id,
            "fitness": fitness,
            "config": candidate.config,
        })
        return fitness

    # ------------------------------------------------------------------
    # Selection hooks
    # ------------------------------------------------------------------
    def select(self, population: List[Candidate]) -> List[Candidate]:
        """Return the top ``selection_size`` candidates by fitness."""
        ranked = sorted(population, key=lambda c: c.fitness if c.fitness is not None else float("inf"))
        return ranked[: self.selection_size]

    # ------------------------------------------------------------------
    def _next_id(self) -> int:
        self._candidate_counter += 1
        return self._candidate_counter

    def _spawn(self, parents: List[Candidate]) -> List[Candidate]:
        """Create a population by mutating ``parents``."""
        new_pop: List[Candidate] = []
        while len(new_pop) < self.population_size:
            parent = random.choice(parents)
            cfg = self.mutate(parent.config)
            new_pop.append(Candidate(id=self._next_id(), config=cfg, parent_id=parent.id))
        return new_pop

    def evolve(self) -> Candidate:
        """Run the evolutionary loop and return the best candidate."""
        base = Candidate(id=self._next_id(), config=self.base_config, parent_id=None)
        population = self._spawn([base])
        population.insert(0, base)
        for _ in range(self.generations):
            with ThreadPoolExecutor(max_workers=self.parallelism) as ex:
                futures = {ex.submit(self.evaluate, c): c for c in population}
                for f in as_completed(futures):
                    f.result()
            parents = self.select(population)
            if _ < self.generations - 1:
                population = self._spawn(parents)
            else:
                population = parents
        best = min(population, key=lambda c: c.fitness if c.fitness is not None else float("inf"))
        return best

    # ------------------------------------------------------------------
    def export_lineage_json(self, path: str) -> None:
        """Serialize lineage information to ``path``."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.lineage, f, indent=2)

    def export_lineage_graph(self, path: str) -> None:
        """Write the evolution graph to ``path`` in GraphML format."""
        nx.write_graphml(self.graph, path)
