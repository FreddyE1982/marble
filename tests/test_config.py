import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config_loader import load_config, create_marble_from_config
from marble_main import MARBLE
from remote_offload import RemoteBrainClient
from torrent_offload import BrainTorrentClient


def test_load_config_defaults():
    cfg = load_config()
    assert 'core' in cfg
    assert cfg['core']['width'] == 30
    assert cfg['core']['representation_size'] == 4
    assert cfg['core']['message_passing_alpha'] == 0.5
    assert cfg['core']['file_tier_path'] == 'data/marble_file_tier.dat'
    assert 'neuronenblitz' in cfg
    assert cfg['brain']['save_threshold'] == 0.05
    assert cfg['meta_controller']['history_length'] == 5
    assert cfg['neuromodulatory_system']['initial']['emotion'] == "neutral"
    assert cfg['remote_client']['url'] == "http://localhost:8001"
    assert cfg['remote_client']['timeout'] == 5.0
    assert cfg['torrent_client']['client_id'] == 'main'
    assert cfg['torrent_client']['buffer_size'] == 10
    assert cfg['brain']['initial_neurogenesis_factor'] == 1.0
    assert cfg['brain']['offload_enabled'] is False
    assert cfg['brain']['torrent_offload_enabled'] is False
    assert cfg['brain']['mutation_rate'] == 0.01
    assert cfg['brain']['mutation_strength'] == 0.05
    assert cfg['brain']['prune_threshold'] == 0.01
    assert cfg['brain']['dream_num_cycles'] == 10
    assert cfg['brain']['dream_interval'] == 5
    assert cfg['brain']['neurogenesis_base_neurons'] == 5
    assert cfg['brain']['neurogenesis_base_synapses'] == 10
    assert cfg['memory_system']['threshold'] == 0.5
    assert cfg['data_compressor']['compression_level'] == 6


def test_create_marble_from_config():
    marble = create_marble_from_config()
    assert isinstance(marble, MARBLE)
    assert marble.brain.meta_controller.history_length == 5
    assert isinstance(marble.brain.remote_client, RemoteBrainClient)
    assert isinstance(marble.brain.torrent_client, BrainTorrentClient)
    assert marble.brain.neurogenesis_factor == 1.0
    assert marble.brain.neurogenesis_base_neurons == 5
    assert marble.brain.neurogenesis_base_synapses == 10
    assert marble.brain.offload_enabled is False
    assert marble.brain.torrent_offload_enabled is False
    assert marble.brain.mutation_rate == 0.01
    assert marble.brain.mutation_strength == 0.05
    assert marble.brain.prune_threshold == 0.01
    assert marble.brain.memory_system.threshold == 0.5
    assert marble.brain.dream_num_cycles == 10
    assert marble.brain.dream_interval == 5
    assert marble.dataloader.compressor.level == 6
    assert marble.core.rep_size == 4
    assert marble.core.params['message_passing_alpha'] == 0.5
