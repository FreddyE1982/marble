import json
from evolution_trainer import EvolutionTrainer, Candidate


def _train(config, steps, device):
    import torch
    x = torch.tensor(config["x"], device=device, dtype=torch.float32)
    target = torch.tensor(1.0, device=device)
    return torch.abs(x - target).item()


def test_mutation_and_selection():
    base = {"x": 0.5, "choice": "a"}
    space = {
        "x": {"type": "float", "min": 0.0, "max": 2.0, "sigma": 0.1},
        "choice": {"type": "categorical", "options": ["a", "b"]},
    }
    trainer = EvolutionTrainer(
        base,
        _train,
        space,
        population_size=4,
        selection_size=2,
        generations=1,
        mutation_rate=1.0,
        steps_per_candidate=1,
        parallelism=1,
    )
    mutated = trainer.mutate(base)
    assert 0.0 <= mutated["x"] <= 2.0
    assert mutated["choice"] in ["a", "b"]
    c1 = Candidate(1, base, fitness=0.2)
    c2 = Candidate(2, base, fitness=0.1)
    selected = trainer.select([c1, c2])
    assert selected[0] is c2
