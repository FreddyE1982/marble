import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import jsonschema
import pytest
import yaml

from config_loader import create_marble_from_config, load_config, validate_global_config
from marble_main import MARBLE
from remote_offload import RemoteBrainClient
from torrent_offload import BrainTorrentClient


def test_load_config_defaults():
    cfg = load_config()
    assert "core" in cfg
    assert cfg["core"]["width"] == 30
    assert cfg["core"]["representation_size"] == 4
    assert cfg["core"]["message_passing_alpha"] == 0.5
    assert cfg["core"]["file_tier_path"] == "data/marble_file_tier.dat"
    assert "neuronenblitz" in cfg
    assert cfg["core"]["init_noise_std"] == 0.0
    assert cfg["core"]["weight_init_min"] == 0.5
    assert cfg["core"]["weight_init_max"] == 1.5
    assert cfg["core"]["weight_init_std"] == 1.0
    assert cfg["core"]["weight_init_type"] == "uniform"
    assert cfg["core"]["mandelbrot_escape_radius"] == 2.0
    assert cfg["core"]["mandelbrot_power"] == 2
    assert cfg["core"]["tier_autotune_enabled"] is True
    assert cfg["core"]["memory_cleanup_interval"] == 60
    assert cfg["core"]["representation_noise_std"] == 0.0
    assert cfg["core"]["workspace_broadcast"] is False
    assert cfg["core"]["gradient_clip_value"] == 1.0
    assert cfg["core"]["synapse_weight_decay"] == 0.0
    assert cfg["core"]["message_passing_iterations"] == 1
    assert cfg["core"]["cluster_algorithm"] == "kmeans"
    assert cfg["brain"]["save_threshold"] == 0.05
    assert cfg["meta_controller"]["history_length"] == 5
    assert cfg["neuromodulatory_system"]["initial"]["emotion"] == "neutral"
    assert cfg["network"]["remote_client"]["url"] == "http://localhost:8001"
    assert cfg["network"]["remote_client"]["timeout"] == 5.0
    assert cfg["network"]["remote_client"]["max_retries"] == 3
    assert cfg["network"]["remote_client"]["backoff_factor"] == 0.5
    assert cfg["remote_hardware"]["grpc"]["max_retries"] == 3
    assert cfg["remote_hardware"]["grpc"]["backoff_factor"] == 0.5
    assert cfg["network"]["torrent_client"]["client_id"] == "main"
    assert cfg["network"]["torrent_client"]["buffer_size"] == 10
    assert cfg["sync"]["interval_ms"] == 1000
    assert cfg["brain"]["initial_neurogenesis_factor"] == 1.0
    assert cfg["brain"]["offload_enabled"] is False
    assert cfg["brain"]["torrent_offload_enabled"] is False
    assert cfg["brain"]["mutation_rate"] == 0.01
    assert cfg["brain"]["mutation_strength"] == 0.05
    assert cfg["brain"]["prune_threshold"] == 0.01
    assert cfg["brain"]["dream_num_cycles"] == 10
    assert cfg["brain"]["dream_interval"] == 5
    assert cfg["brain"]["neurogenesis_base_neurons"] == 5
    assert cfg["brain"]["neurogenesis_base_synapses"] == 10
    assert cfg["brain"]["max_training_epochs"] == 100
    assert cfg["brain"]["memory_cleanup_enabled"] is True
    assert cfg["brain"]["manual_seed"] == 0
    assert cfg["brain"]["log_interval"] == 10
    assert cfg["brain"]["evaluation_interval"] == 1
    assert cfg["brain"]["early_stopping_patience"] == 5
    assert cfg["brain"]["early_stopping_delta"] == 0.001
    assert cfg["brain"]["auto_cluster_interval"] == 5
    assert cfg["brain"]["cluster_method"] == "kmeans"
    assert cfg["brain"]["auto_save_enabled"] is True
    assert cfg["brain"]["offload_threshold"] == 1.0
    assert cfg["brain"]["torrent_offload_threshold"] == 1.0
    assert cfg["brain"]["cluster_high_threshold"] == 1.0
    assert cfg["brain"]["cluster_medium_threshold"] == 0.1
    assert cfg["brain"]["dream_synapse_decay"] == 0.995
    assert cfg["brain"]["neurogenesis_increase_step"] == 0.1
    assert cfg["brain"]["neurogenesis_decrease_step"] == 0.05
    assert cfg["brain"]["max_neurogenesis_factor"] == 3.0
    assert cfg["brain"]["cluster_k"] == 3
    assert cfg["neuronenblitz"]["continue_decay_rate"] == 0.85
    assert cfg["neuronenblitz"]["struct_weight_multiplier1"] == 1.5
    assert cfg["neuronenblitz"]["struct_weight_multiplier2"] == 1.2
    assert cfg["neuronenblitz"]["attention_decay"] == 0.9
    assert cfg["neuronenblitz"]["weight_decay"] == 0.0
    assert cfg["neuronenblitz"]["dropout_probability"] == 0.0
    assert cfg["neuronenblitz"]["exploration_decay"] == 0.99
    assert cfg["neuronenblitz"]["reward_scale"] == 1.0
    assert cfg["neuronenblitz"]["stress_scale"] == 1.0
    assert cfg["neuronenblitz"]["remote_fallback"] is False
    assert cfg["neuronenblitz"]["noise_injection_std"] == 0.0
    assert cfg["neuronenblitz"]["dynamic_attention_enabled"] is True
    assert cfg["neuronenblitz"]["backtrack_depth_limit"] == 10
    assert cfg["neuronenblitz"]["synapse_update_cap"] == 1.0
    assert cfg["neuronenblitz"]["emergent_connection_prob"] == 0.05
    assert cfg["memory_system"]["threshold"] == 0.5
    assert cfg["neuronenblitz"]["max_wander_depth"] == 100
    assert cfg["memory_system"]["consolidation_interval"] == 10
    assert cfg["data_compressor"]["compression_level"] == 6
    assert cfg["data_compressor"]["compression_enabled"] is True
    assert cfg["brain"]["loss_growth_threshold"] == 0.1
    assert cfg["brain"]["dream_cycle_sleep"] == 0.1
    assert cfg["brain"]["dream_replay_buffer_size"] == 100
    assert cfg["brain"]["dream_replay_batch_size"] == 8
    assert cfg["brain"]["dream_replay_weighting"] == "linear"
    assert cfg["brain"]["dream_instant_buffer_size"] == 10
    assert cfg["brain"]["dream_housekeeping_threshold"] == 0.05
    assert cfg["lobe_manager"]["attention_increase_factor"] == 1.05
    assert cfg["lobe_manager"]["attention_decrease_factor"] == 0.95
    assert cfg["network"]["remote_server"]["enabled"] is False
    assert cfg["metrics_visualizer"]["fig_width"] == 10
    assert cfg["metrics_visualizer"]["fig_height"] == 6
    assert cfg["metrics_visualizer"]["track_cpu_usage"] is False
    assert cfg["metrics_visualizer"]["json_log_path"] == "metrics.jsonl"
    assert cfg["brain"]["super_evolution_mode"] is False


def test_create_marble_from_config():
    marble = create_marble_from_config()
    assert isinstance(marble, MARBLE)
    assert marble.brain.meta_controller.history_length == 5
    assert isinstance(marble.brain.remote_client, RemoteBrainClient)
    assert isinstance(marble.brain.torrent_client, BrainTorrentClient)
    assert marble.brain.neurogenesis_factor == 1.0
    assert marble.brain.neurogenesis_base_neurons == 5
    assert marble.brain.neurogenesis_base_synapses == 10
    assert marble.brain.offload_threshold == 1.0
    assert marble.brain.torrent_offload_threshold == 1.0
    assert marble.brain.cluster_high_threshold == 1.0
    assert marble.brain.cluster_medium_threshold == 0.1
    assert marble.brain.dream_synapse_decay == 0.995
    assert marble.brain.neurogenesis_increase_step == 0.1
    assert marble.brain.neurogenesis_decrease_step == 0.05
    assert marble.brain.max_neurogenesis_factor == 3.0
    assert marble.brain.cluster_k == 3
    assert marble.neuronenblitz.continue_decay_rate == 0.85
    assert marble.neuronenblitz.struct_weight_multiplier1 == 1.5
    assert marble.neuronenblitz.struct_weight_multiplier2 == 1.2
    assert marble.neuronenblitz.attention_decay == 0.9
    assert marble.brain.offload_enabled is False
    assert marble.brain.torrent_offload_enabled is False
    assert marble.brain.mutation_rate == 0.01
    assert marble.brain.mutation_strength == 0.05
    assert marble.brain.prune_threshold == 0.01
    assert marble.brain.memory_system.threshold == 0.5
    assert marble.brain.dream_num_cycles == 10
    assert marble.brain.dream_interval == 5
    assert marble.brain.auto_save_interval == 5
    assert marble.brain.max_training_epochs == 100
    assert marble.brain.memory_cleanup_enabled is True
    assert marble.brain.manual_seed == 0
    assert marble.brain.log_interval == 10
    assert marble.brain.evaluation_interval == 1
    assert marble.brain.early_stopping_patience == 5
    assert marble.brain.early_stopping_delta == 0.001
    assert marble.brain.auto_cluster_interval == 5
    assert marble.brain.cluster_method == "kmeans"
    assert marble.brain.auto_save_enabled is True
    assert marble.neuronenblitz.max_wander_depth == 100
    assert marble.dataloader.compressor.level == 6
    assert marble.core.rep_size == 4
    assert marble.core.params["message_passing_alpha"] == 0.5
    assert marble.core.params["workspace_broadcast"] is False
    assert marble.core.synapse_weight_decay == 0.0
    assert marble.brain.loss_growth_threshold == 0.1
    assert marble.brain.dream_cycle_sleep == 0.1
    assert marble.brain.dream_buffer.capacity == 100
    assert marble.brain.dream_replay_batch_size == 8
    assert marble.brain.dream_buffer.weighting == "linear"
    assert marble.brain.dream_buffer.instant_capacity == 10
    assert marble.brain.dream_buffer.housekeeping_threshold == 0.05
    assert marble.brain.lobe_manager.attention_increase_factor == 1.05
    assert marble.brain.lobe_manager.attention_decrease_factor == 0.95
    assert marble.brain.super_evolution_mode is False
    assert marble.brain.super_evo_controller is None
    assert hasattr(marble, "remote_server") is False


def test_remote_server_start(tmp_path):
    cfg_path = tmp_path / "cfg.yaml"
    cfg = load_config()
    cfg.setdefault("network", {})["remote_server"] = {
        "enabled": True,
        "host": "localhost",
        "port": 8123,
        "remote_url": None,
        "auth_token": "abc",
    }
    with open(cfg_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f)
    marble = create_marble_from_config(str(cfg_path))
    assert hasattr(marble, "remote_server")
    marble.remote_server.stop()


def test_synapse_update_cap_configurable(tmp_path):
    cfg_path = tmp_path / "cfg.yaml"
    cfg = load_config()
    cfg["neuronenblitz"]["synapse_update_cap"] = 0.5
    with open(cfg_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f)
    marble = create_marble_from_config(str(cfg_path))
    assert marble.neuronenblitz.synapse_update_cap == 0.5


def test_new_nb_parameters_configurable(tmp_path):
    cfg_path = tmp_path / "cfg.yaml"
    cfg = load_config()
    nb = cfg["neuronenblitz"]
    nb["weight_limit"] = 123.0
    nb["wander_cache_size"] = 7
    nb["rmsprop_beta"] = 0.8
    nb["grad_epsilon"] = 1e-6
    with open(cfg_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f)
    marble = create_marble_from_config(str(cfg_path))
    assert marble.neuronenblitz._weight_limit == 123.0
    assert marble.neuronenblitz._cache_max_size == 7
    assert marble.neuronenblitz._rmsprop_beta == 0.8
    assert marble.neuronenblitz._grad_epsilon == 1e-6


def test_global_workspace_config(tmp_path):
    cfg_path = tmp_path / "cfg.yaml"
    cfg = load_config()
    cfg["global_workspace"] = {"enabled": True, "capacity": 2}
    with open(cfg_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f)
    marble = create_marble_from_config(str(cfg_path))
    import global_workspace

    assert hasattr(marble.neuronenblitz, "global_workspace")
    assert global_workspace.workspace.queue.maxlen == 2


def test_invalid_config_raises(tmp_path):
    cfg = load_config()
    cfg["brain"]["early_stopping_patience"] = -1
    cfg_path = tmp_path / "bad.yaml"
    with open(cfg_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f)
    with pytest.raises(jsonschema.ValidationError):
        load_config(str(cfg_path))


def test_validate_global_config_missing_section(tmp_path):
    cfg = load_config()
    del cfg["core"]
    with pytest.raises(ValueError):
        validate_global_config(cfg)


def test_partial_config_merges_defaults(tmp_path):
    partial = {"core": {"width": 5}}
    cfg_path = tmp_path / "partial.yaml"
    with open(cfg_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(partial, f)
    cfg = load_config(str(cfg_path))
    default = load_config()
    assert cfg["core"]["width"] == 5
    assert cfg["core"]["height"] == default["core"]["height"]


def test_dataloader_round_trip_config(tmp_path):
    cfg = load_config()
    dl_cfg = cfg.setdefault("dataloader", {})
    dl_cfg["enable_round_trip_check"] = True
    dl_cfg["round_trip_penalty"] = 0.7
    dl_cfg["track_metadata"] = False
    cfg_path = tmp_path / "rt.yaml"
    with open(cfg_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f)
    marble = create_marble_from_config(str(cfg_path))
    assert marble.dataloader.enable_round_trip_check is True
    assert marble.dataloader.round_trip_penalty == 0.7
    assert marble.dataloader.track_metadata is False
