import os, sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from marble_core import Core, DataLoader
from marble_neuronenblitz import Neuronenblitz
from marble_brain import Brain
from tests.test_core_functions import minimal_params
from tqdm import tqdm as std_tqdm


def test_super_evolution_records_metrics():
    params = minimal_params()
    core = Core(params)
    nb = Neuronenblitz(core)
    import marble_brain as mb

    mb.tqdm = std_tqdm
    brain = Brain(core, nb, DataLoader(), super_evolution_mode=True)
    examples = [(0.1, 0.2), (0.2, 0.3)]
    brain.train(examples, epochs=1, validation_examples=examples)
    assert brain.super_evo_controller.history


def test_super_evolution_affects_all_parameters():
    params = minimal_params()
    core = Core(params)
    nb = Neuronenblitz(core)
    import marble_brain as mb

    mb.tqdm = std_tqdm
    brain = Brain(core, nb, DataLoader(), super_evolution_mode=True)
    brain.lobe_manager.genesis([0])
    brain.core.neurons[0].attention_score = 1.0
    brain.lobe_manager.update_attention()

    brain.meta_controller.adjustment = 0.5
    brain.neuromodulatory_system.signals["arousal"] = 1.0
    brain.memory_system.threshold = 0.5

    orig_values = (
        brain.core.params["vram_limit_mb"],
        brain.neuronenblitz.learning_rate,
        brain.mutation_rate,
        brain.meta_controller.adjustment,
        brain.neuromodulatory_system.signals["arousal"],
        brain.memory_system.threshold,
        brain.dataloader.compressor.level,
        brain.lobe_manager.attention_increase_factor,
    )

    brain.super_evo_controller.record_metrics(0.5, 1.0)
    brain.super_evo_controller.record_metrics(1.0, 1.1)

    assert brain.core.params["vram_limit_mb"] != orig_values[0]
    assert brain.neuronenblitz.learning_rate != orig_values[1]
    assert brain.mutation_rate != orig_values[2]
    assert brain.meta_controller.adjustment != orig_values[3]
    assert brain.neuromodulatory_system.signals["arousal"] != orig_values[4]
    assert brain.memory_system.threshold != orig_values[5]
    assert brain.dataloader.compressor.level != orig_values[6]
    assert brain.lobe_manager.attention_increase_factor != orig_values[7]


def test_super_evolution_skips_blocked_parameters():
    params = minimal_params()
    core = Core(params)
    nb = Neuronenblitz(core)
    import marble_brain as mb

    mb.tqdm = std_tqdm
    brain = Brain(core, nb, DataLoader(), super_evolution_mode=True)
    brain.lobe_manager.genesis([0])
    brain.core.neurons[0].attention_score = 1.0
    brain.lobe_manager.update_attention()

    orig = {
        "auto_save_interval": brain.auto_save_interval,
        "benchmark_interval": brain.benchmark_interval,
        "loss_growth_threshold": brain.loss_growth_threshold,
        "metrics_history_size": brain.metrics_history_size,
        "last_val_loss": brain.last_val_loss,
        "tier_decision_vram": brain.tier_decision_params["vram_usage_threshold"],
        "tier_decision_ram": brain.tier_decision_params["ram_usage_threshold"],
        "gradient_clip_value": brain.core.gradient_clip_value,
        "random_seed": brain.core.params["random_seed"],
        "weight_init_min": brain.core.weight_init_min,
        "weight_init_max": brain.core.weight_init_max,
        "loss_scale": brain.neuronenblitz.loss_scale,
        "representation_size": brain.core.params["representation_size"],
        "manual_seed": brain.manual_seed,
        "model_name": brain.model_name,
        "checkpoint_format": brain.checkpoint_format,
        "super_evolution_mode": brain.super_evolution_mode,
    }

    brain.super_evo_controller.record_metrics(0.5, 1.0)
    brain.super_evo_controller.record_metrics(1.0, 1.1)

    assert brain.auto_save_interval == orig["auto_save_interval"]
    assert brain.benchmark_interval == orig["benchmark_interval"]
    assert brain.loss_growth_threshold == orig["loss_growth_threshold"]
    assert brain.metrics_history_size == orig["metrics_history_size"]
    assert brain.last_val_loss == orig["last_val_loss"]
    assert brain.core.gradient_clip_value == orig["gradient_clip_value"]
    assert brain.core.params["random_seed"] == orig["random_seed"]
    assert brain.core.weight_init_min == orig["weight_init_min"]
    assert brain.core.weight_init_max == orig["weight_init_max"]
    assert brain.neuronenblitz.loss_scale == orig["loss_scale"]
    assert brain.core.params["representation_size"] == orig["representation_size"]
    assert brain.manual_seed == orig["manual_seed"]
    assert brain.model_name == orig["model_name"]
    assert brain.checkpoint_format == orig["checkpoint_format"]
    assert brain.super_evolution_mode == orig["super_evolution_mode"]
    assert (
        brain.tier_decision_params["vram_usage_threshold"] == orig["tier_decision_vram"]
    )
    assert (
        brain.tier_decision_params["ram_usage_threshold"] == orig["tier_decision_ram"]
    )
