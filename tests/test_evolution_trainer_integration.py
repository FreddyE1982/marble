import json
from pathlib import Path

from evolution_trainer import EvolutionTrainer


def _train(config, steps, device):
    import torch
    x = torch.tensor(config["x"], device=device, dtype=torch.float32)
    target = torch.tensor(1.0, device=device)
    loss = torch.abs(x - target)
    return loss.item()


def test_evolution_over_generations(tmp_path):
    base = {"x": 0.0, "choice": "a"}
    space = {
        "x": {"type": "float", "min": 0.0, "max": 2.0, "sigma": 0.2},
        "choice": {"type": "categorical", "options": ["a", "b"]},
    }
    trainer = EvolutionTrainer(
        base,
        _train,
        space,
        population_size=4,
        selection_size=2,
        generations=2,
        steps_per_candidate=1,
        mutation_rate=1.0,
        parallelism=2,
    )
    best = trainer.evolve()
    assert isinstance(best.fitness, float)
    json_path = tmp_path / "lineage.json"
    graph_path = tmp_path / "lineage.graphml"
    trainer.export_lineage_json(str(json_path))
    trainer.export_lineage_graph(str(graph_path))
    assert json_path.is_file()
    assert graph_path.is_file()
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert any(entry["parent"] is not None for entry in data)
