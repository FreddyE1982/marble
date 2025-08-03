from pathlib import Path

import yaml

from config_schema import validate_config_schema
from marble_core import MemorySystem, TIER_REGISTRY
from marble_main import MARBLE
from meta_parameter_controller import MetaParameterController
from neuromodulatory_system import NeuromodulatorySystem
from plugin_system import load_plugins
from remote_hardware import load_remote_tier_plugin
from remote_offload import RemoteBrainClient, RemoteBrainServer
from torrent_offload import BrainTorrentClient, BrainTorrentTracker

import tensor_backend as tb

DEFAULT_CONFIG_FILE = Path(__file__).resolve().parent / "config.yaml"


def _deep_update(base: dict, updates: dict) -> None:
    """Recursively update ``base`` with values from ``updates``."""
    for key, val in updates.items():
        if isinstance(val, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], val)
        else:
            base[key] = val


def load_config(path: str | None = None) -> dict:
    """Load configuration from a YAML file and merge with defaults."""
    with open(DEFAULT_CONFIG_FILE, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if path is not None:
        cfg_path = Path(path)
        if cfg_path.is_file():
            with open(cfg_path, "r", encoding="utf-8") as f:
                overrides = yaml.safe_load(f) or {}
            _deep_update(data, overrides)
    validate_config_schema(data)
    return data


def validate_global_config(cfg: dict) -> None:
    """Perform basic sanity checks on the full configuration."""

    required_sections = [
        "core",
        "neuronenblitz",
        "brain",
        "dataloader",
        "memory_system",
    ]
    for sec in required_sections:
        if sec not in cfg:
            raise ValueError(f"Missing required config section: {sec}")


def create_marble_from_config(
    path: str | None = None, *, overrides: dict | None = None
) -> MARBLE:
    """Create a :class:`MARBLE` instance from a YAML configuration.

    Parameters
    ----------
    path:
        Optional path to the YAML configuration file. If ``None`` the default
        ``config.yaml`` next to this module is used.
    overrides:
        Optional dictionary of parameters that override those loaded from the
        configuration file. Nested dictionaries are merged recursively.
    """
    cfg = load_config(path)
    if overrides:
        _deep_update(cfg, overrides)
    validate_global_config(cfg)

    plugin_dirs = cfg.get("plugins", [])
    if plugin_dirs:
        load_plugins(plugin_dirs)

    core_params = cfg.get("core", {})
    tb.set_backend(core_params.get("backend", "numpy"))
    qbits = core_params.get("quantization_bits", 0)
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
    memory_system = MemorySystem(
        long_term_path,
        threshold=threshold,
        consolidation_interval=consolidation_interval,
    )

    hybrid_memory_params = cfg.get("hybrid_memory", {})

    # Data compressor
    compressor_cfg = cfg.get("data_compressor", {})
    compression_level = compressor_cfg.get("compression_level", 6)
    compression_enabled = compressor_cfg.get("compression_enabled", True)
    dataloader_cfg = cfg.get("dataloader", {})
    tensor_dtype = dataloader_cfg.get("tensor_dtype", "uint8")
    dataloader_params = {
        "compression_level": compression_level,
        "compression_enabled": compression_enabled,
        "tensor_dtype": tensor_dtype,
        "track_metadata": dataloader_cfg.get("track_metadata", True),
        "enable_round_trip_check": dataloader_cfg.get("enable_round_trip_check", False),
        "round_trip_penalty": dataloader_cfg.get("round_trip_penalty", 0.0),
    }
    for key in ["tokenizer_type", "tokenizer_json", "tokenizer_vocab_size"]:
        if key in dataloader_cfg:
            dataloader_params[key] = dataloader_cfg[key]

    autograd_params = cfg.get("autograd", {})
    gw_cfg = cfg.get("global_workspace", {})
    ac_cfg = cfg.get("attention_codelets", {})
    pytorch_challenge_params = cfg.get("pytorch_challenge", {})
    gpt_cfg = cfg.get("gpt", {})

    brain_params.update(
        {
            "neuromodulatory_system": neuromod_system,
            "meta_controller": meta_controller,
            "memory_system": memory_system,
            "hybrid_memory_params": hybrid_memory_params,
            "initial_neurogenesis_factor": initial_neurogenesis_factor,
            "dream_num_cycles": dream_num_cycles,
            "dream_interval": dream_interval,
            "neurogenesis_base_neurons": neuro_base_neurons,
            "neurogenesis_base_synapses": neuro_base_synapses,
            "super_evolution_mode": super_evolution_mode,
            "checkpoint_format": brain_params.get("checkpoint_format", "pickle"),
            "checkpoint_compress": brain_params.get("checkpoint_compress", False),
        }
    )

    network_cfg = cfg.get("network", {})
    rh_cfg = cfg.get("remote_hardware", {})
    remote_client = None
    remote_cfg = network_cfg.get("remote_client", {})
    if isinstance(remote_cfg, dict) and remote_cfg.get("url"):
        remote_client = RemoteBrainClient(
            remote_cfg["url"],
            timeout=remote_cfg.get("timeout", 5.0),
            max_retries=remote_cfg.get("max_retries", 3),
            auth_token=remote_cfg.get("auth_token"),
        )

    remote_server = None
    server_cfg = network_cfg.get("remote_server", {})
    if isinstance(server_cfg, dict) and server_cfg.get("enabled", False):
        remote_server = RemoteBrainServer(
            host=server_cfg.get("host", "localhost"),
            port=server_cfg.get("port", 8000),
            remote_url=server_cfg.get("remote_url"),
            auth_token=server_cfg.get("auth_token"),
        )
        remote_server.start()

    remote_tier = None
    if isinstance(rh_cfg, dict) and rh_cfg.get("tier_plugin"):
        plugin = rh_cfg["tier_plugin"]
        kwargs = rh_cfg.get("grpc", {})
        remote_tier = load_remote_tier_plugin(plugin, **kwargs)
        TIER_REGISTRY[remote_tier.name] = remote_tier

    torrent_client = None
    torrent_cfg = network_cfg.get("torrent_client", {})
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
        autograd_params=autograd_params,
        pytorch_challenge_params=pytorch_challenge_params,
        hybrid_memory_params=hybrid_memory_params,
    )
    if gw_cfg.get("enabled", False):
        from global_workspace import activate as activate_global_workspace

        activate_global_workspace(
            marble.neuronenblitz if hasattr(marble, "neuronenblitz") else None,
            capacity=gw_cfg.get("capacity", 100),
        )
    if ac_cfg.get("enabled", False):
        from attention_codelets import activate as activate_codelets

        activate_codelets(coalition_size=ac_cfg.get("coalition_size", 1))
    if qbits:
        from model_quantization import quantize_core_weights

        quantize_core_weights(int(qbits))
    if remote_server is not None:
        marble.remote_server = remote_server

    if gpt_cfg.get("enabled", False) and gpt_cfg.get("dataset_path"):
        from advanced_gpt import load_text_dataset, train_advanced_gpt

        dataset, _ = load_text_dataset(
            gpt_cfg["dataset_path"],
            gpt_cfg.get("vocab_size", 50),
            gpt_cfg.get("block_size", 8),
        )
        marble.gpt_model, _ = train_advanced_gpt(
            dataset,
            vocab_size=gpt_cfg.get("vocab_size", 50),
            block_size=gpt_cfg.get("block_size", 8),
            num_layers=gpt_cfg.get("num_layers", 2),
            num_heads=gpt_cfg.get("num_heads", 2),
            hidden_dim=gpt_cfg.get("hidden_dim", 64),
            epochs=gpt_cfg.get("num_train_steps", 1),
            lr=gpt_cfg.get("learning_rate", 1e-3),
            batch_size=gpt_cfg.get("batch_size", 1),
        )
    return marble
