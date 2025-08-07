import yaml
import evolution_trainer as et
import config_loader


def _train(cfg, steps, device):
    return 0.0


def test_run_evolution_defaults_from_config(tmp_path, monkeypatch):
    cfg = config_loader.load_config()
    cfg['evolution'] = {
        'population_size': 3,
        'selection_size': 1,
        'generations': 2,
        'steps_per_candidate': 1,
        'mutation_rate': 0.5,
        'parallelism': 1,
    }
    cfg_path = tmp_path / 'cfg.yaml'
    cfg_path.write_text(yaml.safe_dump(cfg))
    monkeypatch.setattr(config_loader, 'DEFAULT_CONFIG_FILE', cfg_path)

    captured = {}
    orig_init = et.EvolutionTrainer.__init__

    def spy_init(self, base_config, train_func, mutation_space, population_size, selection_size, generations, *, mutation_rate=0.1, steps_per_candidate=5, parallelism=None):
        captured['population_size'] = population_size
        captured['selection_size'] = selection_size
        captured['generations'] = generations
        captured['steps_per_candidate'] = steps_per_candidate
        captured['mutation_rate'] = mutation_rate
        captured['parallelism'] = parallelism
        orig_init(self, base_config, train_func, mutation_space, population_size, selection_size, generations, mutation_rate=mutation_rate, steps_per_candidate=steps_per_candidate, parallelism=parallelism)

    monkeypatch.setattr(et.EvolutionTrainer, '__init__', spy_init)
    monkeypatch.setattr(et.EvolutionTrainer, 'evolve', lambda self: et.Candidate(0, {}))

    et.run_evolution({'x': 0.0}, _train, {'x': {'type': 'float', 'min': 0.0, 'max': 1.0, 'sigma': 0.1}})

    assert captured == {
        'population_size': 3,
        'selection_size': 1,
        'generations': 2,
        'steps_per_candidate': 1,
        'mutation_rate': 0.5,
        'parallelism': 1,
    }
