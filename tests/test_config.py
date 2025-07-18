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


def test_create_marble_from_config():
    marble = create_marble_from_config()
    assert isinstance(marble, MARBLE)
    assert marble.brain.meta_controller.history_length == 5
    assert isinstance(marble.brain.remote_client, RemoteBrainClient)
    assert isinstance(marble.brain.torrent_client, BrainTorrentClient)
    assert marble.brain.neurogenesis_factor == 1.0
    assert marble.brain.offload_enabled is False
    assert marble.brain.torrent_offload_enabled is False
    assert marble.brain.mutation_rate == 0.01
    assert marble.brain.mutation_strength == 0.05
    assert marble.brain.prune_threshold == 0.01
