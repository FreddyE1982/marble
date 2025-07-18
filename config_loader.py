from pathlib import Path
import yaml

from marble_main import MARBLE
from neuromodulatory_system import NeuromodulatorySystem
from meta_parameter_controller import MetaParameterController
from marble_core import MemorySystem
from remote_offload import RemoteBrainClient, RemoteBrainServer
from torrent_offload import BrainTorrentClient, BrainTorrentTracker

DEFAULT_CONFIG_FILE = Path(__file__).resolve().parent / "config.yaml"

def load_config(path: str | None = None) -> dict:
    """Load configuration from a YAML file."""
    cfg_path = Path(path) if path is not None else DEFAULT_CONFIG_FILE
    with open(cfg_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data


def create_marble_from_config(path: str | None = None) -> MARBLE:
    """Create a :class:`MARBLE` instance from a YAML configuration."""
    cfg = load_config(path)

    core_params = cfg.get("core", {})
    nb_params = cfg.get("neuronenblitz", {})
    brain_params = cfg.get("brain", {})
    initial_neurogenesis_factor = brain_params.pop("initial_neurogenesis_factor", 1.0)
    dream_num_cycles = brain_params.pop("dream_num_cycles", 10)
    dream_interval = brain_params.pop("dream_interval", 5)
    neuro_base_neurons = brain_params.pop("neurogenesis_base_neurons", 5)
    neuro_base_synapses = brain_params.pop("neurogenesis_base_synapses", 10)
    super_evolution_mode = brain_params.pop("super_evolution_mode", False)

    formula = cfg.get("formula")
    formula_num_neurons = cfg.get("formula_num_neurons", 100)

    # Meta-parameter controller
    mc_cfg = cfg.get("meta_controller", {})
    meta_controller = MetaParameterController(
        history_length=mc_cfg.get("history_length", 5),
        adjustment=mc_cfg.get("adjustment", 0.5),
        min_threshold=mc_cfg.get("min_threshold", 1.0),
        max_threshold=mc_cfg.get("max_threshold", 20.0),
    )

    # Neuromodulatory system
    ns_init = cfg.get("neuromodulatory_system", {}).get("initial", {})
    neuromod_system = NeuromodulatorySystem(initial=ns_init)

    # Memory system
    memory_cfg = cfg.get("memory_system", {})
    long_term_path = memory_cfg.get("long_term_path", "long_term_memory.pkl")
    threshold = memory_cfg.get("threshold", 0.5)
    consolidation_interval = memory_cfg.get("consolidation_interval", 10)
    memory_system = MemorySystem(long_term_path, threshold=threshold, consolidation_interval=consolidation_interval)

    # Data compressor
    compressor_cfg = cfg.get("data_compressor", {})
    compression_level = compressor_cfg.get("compression_level", 6)
    dataloader_params = {"compression_level": compression_level}

    brain_params.update({
        "neuromodulatory_system": neuromod_system,
        "meta_controller": meta_controller,
        "memory_system": memory_system,
        "initial_neurogenesis_factor": initial_neurogenesis_factor,
        "dream_num_cycles": dream_num_cycles,
        "dream_interval": dream_interval,
        "neurogenesis_base_neurons": neuro_base_neurons,
        "neurogenesis_base_synapses": neuro_base_synapses,
        "super_evolution_mode": super_evolution_mode,
    })

    remote_client = None
    remote_cfg = cfg.get("remote_client", {})
    if isinstance(remote_cfg, dict) and remote_cfg.get("url"):
        remote_client = RemoteBrainClient(
            remote_cfg["url"],
            timeout=remote_cfg.get("timeout", 5.0),
            max_retries=remote_cfg.get("max_retries", 3),
        )

    remote_server = None
    server_cfg = cfg.get("remote_server", {})
    if isinstance(server_cfg, dict) and server_cfg.get("enabled", False):
        remote_server = RemoteBrainServer(
            host=server_cfg.get("host", "localhost"),
            port=server_cfg.get("port", 8000),
            remote_url=server_cfg.get("remote_url"),
        )
        remote_server.start()

    torrent_client = None
    torrent_cfg = cfg.get("torrent_client", {})
    if isinstance(torrent_cfg, dict) and torrent_cfg.get("client_id"):
        tracker = BrainTorrentTracker()
        torrent_client = BrainTorrentClient(
            torrent_cfg["client_id"],
            tracker,
            buffer_size=torrent_cfg.get("buffer_size", 10),
            heartbeat_interval=torrent_cfg.get("heartbeat_interval", 30),
        )
        torrent_client.connect()

    mv_params = cfg.get("metrics_visualizer", {})
    dashboard_params = cfg.get("metrics_dashboard", {})

    marble = MARBLE(
        core_params,
        formula=formula,
        formula_num_neurons=formula_num_neurons,
        nb_params=nb_params,
        brain_params=brain_params,
        dataloader_params=dataloader_params,
        remote_client=remote_client,
        torrent_client=torrent_client,
        mv_params=mv_params,
        dashboard_params=dashboard_params,
    )
    if remote_server is not None:
        marble.remote_server = remote_server
    return marble
