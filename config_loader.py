from pathlib import Path

import requests
import torch
import yaml

import tensor_backend as tb
from config_schema import validate_config_schema
from logging_utils import configure_structured_logging
from marble_core import TIER_REGISTRY, MemorySystem
from marble_main import MARBLE
from meta_parameter_controller import MetaParameterController
from neuromodulatory_system import NeuromodulatorySystem
from plugin_system import load_plugins
from remote_hardware import load_remote_tier_plugin
from remote_offload import RemoteBrainClient, RemoteBrainServer
from torrent_offload import BrainTorrentClient, BrainTorrentTracker

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
    nb = data.setdefault("neuronenblitz", {})
    nb.setdefault("attention", {}).setdefault("dynamic_span", False)
    meta = data.setdefault("meta", {})
    meta.setdefault("rate", 0.5)
    meta.setdefault("window", 5)
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

    log_cfg = cfg.get("logging", {})
    configure_structured_logging(
        log_cfg.get("structured", False),
        log_cfg.get("log_file"),
        level=log_cfg.get("level", "INFO"),
        format=log_cfg.get("format", "%(levelname)s:%(name)s:%(message)s"),
        datefmt=log_cfg.get("datefmt", "%Y-%m-%d %H:%M:%S"),
        propagate=log_cfg.get("propagate", False),
        rotate=log_cfg.get("rotate", False),
        max_bytes=log_cfg.get("max_bytes", 10_000_000),
        backup_count=log_cfg.get("backup_count", 5),
        encoding=log_cfg.get("encoding", "utf-8"),
    )

    plugin_dirs = cfg.get("plugins", [])
    if plugin_dirs:
        load_plugins(plugin_dirs)

    # Configure asynchronous task scheduler
    from scheduler_plugins import configure_scheduler

    sched_cfg = cfg.get("scheduler", {})
    configure_scheduler(sched_cfg.get("plugin", "thread"))

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

    dataset_path = cfg.get("dataset", {}).get("source")
    if nb_params.get("auto_update") and dataset_path:
        nb_params.setdefault("dataset_path", dataset_path)

    formula = cfg.get("formula")
    formula_num_neurons = cfg.get("formula_num_neurons", 100)

    # Meta-parameter controller
    meta_defaults = cfg.get("meta", {})
    mc_cfg = cfg.get("meta_controller", {})
    meta_controller = MetaParameterController(
        history_length=mc_cfg.get("history_length", meta_defaults.get("window", 5)),
        adjustment=mc_cfg.get("adjustment", meta_defaults.get("rate", 0.5)),
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
    sparse_threshold = compressor_cfg.get("sparse_threshold")
    comp_qbits = compressor_cfg.get("quantization_bits", qbits)
    dataloader_cfg = cfg.get("dataloader", {})
    tensor_dtype = dataloader_cfg.get("tensor_dtype", "uint8")
    dataloader_params = {
        "compression_level": compression_level,
        "compression_enabled": compression_enabled,
        "tensor_dtype": tensor_dtype,
        "track_metadata": dataloader_cfg.get("track_metadata", True),
        "enable_round_trip_check": dataloader_cfg.get("enable_round_trip_check", False),
        "round_trip_penalty": dataloader_cfg.get("round_trip_penalty", 0.0),
        "quantization_bits": comp_qbits,
    }
    if sparse_threshold is not None:
        dataloader_params["sparse_threshold"] = sparse_threshold
    for key in ["tokenizer_type", "tokenizer_json", "tokenizer_vocab_size"]:
        if key in dataloader_cfg:
            dataloader_params[key] = dataloader_cfg[key]

    autograd_params = cfg.get("autograd", {})
    gw_cfg = cfg.get("global_workspace", {})
    ac_cfg = cfg.get("attention_codelets", {})
    ci_cfg = cfg.get("conceptual_integration", {})
    ul_cfg = cfg.get("unified_learning", {})
    tom_cfg = cfg.get("theory_of_mind", {})
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
            backoff_factor=remote_cfg.get("backoff_factor", 0.5),
            connect_retry_interval=remote_cfg.get("connect_retry_interval", 5.0),
            heartbeat_timeout=remote_cfg.get("heartbeat_timeout", 10.0),
            ssl_verify=remote_cfg.get("ssl_verify", True),
        )
        try:
            remote_client.connect()
        except requests.RequestException:
            pass

    remote_server = None
    server_cfg = network_cfg.get("remote_server", {})
    if isinstance(server_cfg, dict) and server_cfg.get("enabled", False):
        remote_server = RemoteBrainServer(
            host=server_cfg.get("host", "localhost"),
            port=server_cfg.get("port", 8000),
            remote_url=server_cfg.get("remote_url"),
            auth_token=server_cfg.get("auth_token"),
            ssl_enabled=server_cfg.get("ssl_enabled", False),
            ssl_cert_file=server_cfg.get("ssl_cert_file"),
            ssl_key_file=server_cfg.get("ssl_key_file"),
            compression_level=server_cfg.get("compression_level", 6),
            compression_enabled=server_cfg.get("compression_enabled", True),
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
    live_cfg = cfg.get("live_kuzu", {})
    kuzu_tracker = None
    if live_cfg.get("enabled", False):
        from experiment_tracker import KuzuExperimentTracker, attach_tracker_to_events
        from topology_kuzu import TopologyKuzuTracker

        db_path = live_cfg.get("db_path", "live.kuzu")
        kuzu_tracker = KuzuExperimentTracker(db_path)
        mv_params = dict(mv_params)
        mv_params["tracker"] = kuzu_tracker
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
    if kuzu_tracker is not None:
        from event_bus import PROGRESS_EVENT

        marble.experiment_tracker = kuzu_tracker
        marble.topology_tracker = TopologyKuzuTracker(marble.core, db_path)
        marble._tracker_detach = attach_tracker_to_events(
            kuzu_tracker, events=[PROGRESS_EVENT]
        )

    pc_cfg = cfg.get("predictive_coding", {})
    if pc_cfg:
        import predictive_coding

        marble.core.predictive_coding = predictive_coding.activate(
            marble.neuronenblitz if hasattr(marble, "neuronenblitz") else None,
            num_layers=int(pc_cfg.get("num_layers", 2)),
            latent_dim=int(pc_cfg.get("latent_dim", marble.core.rep_size)),
            learning_rate=float(pc_cfg.get("learning_rate", 0.001)),
        )

    # Optional tool manager instantiation
    tool_cfg = cfg.get("tool_manager", {})
    if tool_cfg.get("enabled"):
        from tool_manager_plugin import ToolManagerPlugin

        manager = ToolManagerPlugin(
            tools=tool_cfg.get("tools", {}),
            policy=tool_cfg.get("policy", "heuristic"),
            mode=tool_cfg.get("mode", "direct"),
            agent_id=tool_cfg.get("agent_id", "tool_manager"),
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        manager.initialise(device, marble)
        marble.tool_manager = manager

    # Background tensor synchronisation
    sync_cfg = cfg.get("sync", {})
    if "interval_ms" in sync_cfg:
        from tensor_sync_service import TensorSyncService

        marble.tensor_sync_service = TensorSyncService(
            interval_ms=int(sync_cfg.get("interval_ms", 1000))
        )
    if ul_cfg.get("enabled"):
        from unified_learning import UnifiedLearner

        learners = ul_cfg.get("learners", {})
        marble.unified_learner = UnifiedLearner(
            marble.core,
            marble.neuronenblitz if hasattr(marble, "neuronenblitz") else marble,
            learners,
            gating_hidden=ul_cfg.get("gating_hidden", 8),
            log_path=ul_cfg.get("log_path"),
            plugin_dirs=ul_cfg.get("plugin_dirs"),
        )
    if gw_cfg.get("enabled", False):
        from global_workspace import activate as activate_global_workspace

        activate_global_workspace(
            marble.neuronenblitz if hasattr(marble, "neuronenblitz") else None,
            capacity=gw_cfg.get("capacity", 100),
        )
    if ac_cfg.get("enabled", False):
        from attention_codelets import activate as activate_codelets

        activate_codelets(
            coalition_size=ac_cfg.get("coalition_size", 1),
            salience_weight=core_params.get("salience_weight", 1.0),
        )
    if ci_cfg.get("enabled", False):
        from conceptual_integration import ConceptualIntegrationLearner

        nb_inst = marble.neuronenblitz if hasattr(marble, "neuronenblitz") else marble
        marble.conceptual_integration = ConceptualIntegrationLearner(
            marble.core,
            nb_inst,
            blend_probability=ci_cfg.get("blend_probability", 0.1),
            similarity_threshold=ci_cfg.get("similarity_threshold", 0.3),
        )
    if tom_cfg:
        from theory_of_mind import activate as activate_tom

        nb_inst = marble.neuronenblitz if hasattr(marble, "neuronenblitz") else marble
        marble.theory_of_mind = activate_tom(
            nb_inst,
            hidden_size=tom_cfg.get("hidden_size", 16),
            num_layers=tom_cfg.get("num_layers", 1),
            prediction_horizon=tom_cfg.get("prediction_horizon", 1),
            memory_slots=tom_cfg.get("memory_slots", 16),
            attention_hops=tom_cfg.get("attention_hops", 1),
            mismatch_threshold=tom_cfg.get("mismatch_threshold", 0.1),
        )
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
        marble.gpt_model, _, _ = train_advanced_gpt(
            dataset,
            vocab_size=gpt_cfg.get("vocab_size", 50),
            block_size=gpt_cfg.get("block_size", 8),
            num_layers=gpt_cfg.get("num_layers", 2),
            num_heads=gpt_cfg.get("num_heads", 2),
            hidden_dim=gpt_cfg.get("hidden_dim", 64),
            epochs=gpt_cfg.get("num_train_steps", 1),
            lr=gpt_cfg.get("learning_rate", 1e-3),
            batch_size=gpt_cfg.get("batch_size", 1),
            distill_alpha=cfg.get("meta_learning", {}).get("distill_alpha", 0.0),
        )

    rl_cfg = cfg.get("reinforcement_learning", {})
    if rl_cfg.get("enabled", False):
        from reinforcement_learning import (
            GridWorld,
            MarblePolicyGradientAgent,
            MarbleQLearningAgent,
            train_gridworld,
            train_policy_gradient,
        )

        env = GridWorld()
        algorithm = rl_cfg.get("algorithm", "q_learning").lower()
        episodes = int(rl_cfg.get("episodes", 1))
        max_steps = int(rl_cfg.get("max_steps", 50))
        seed = rl_cfg.get("seed", 0)
        if algorithm == "policy_gradient":
            agent = MarblePolicyGradientAgent(
                marble.core,
                marble.neuronenblitz,
                hidden_dim=int(rl_cfg.get("policy_hidden_dim", 16)),
                lr=rl_cfg.get("policy_lr", 0.01),
            )
            train_policy_gradient(agent, env, episodes, max_steps=max_steps, seed=seed)
        else:
            agent = MarbleQLearningAgent(
                marble.core,
                marble.neuronenblitz,
                discount=rl_cfg.get("discount_factor", 0.9),
                epsilon=rl_cfg.get("epsilon_start", 1.0),
                epsilon_decay=rl_cfg.get("epsilon_decay", 0.95),
                min_epsilon=rl_cfg.get("epsilon_min", 0.1),
                double_q=rl_cfg.get("double_q", False),
                learning_rate=rl_cfg.get("learning_rate", 0.1),
            )
            train_gridworld(agent, env, episodes, max_steps=max_steps, seed=seed)
        marble.rl_agent = agent

    adv_cfg = cfg.get("adversarial_learning", {})
    if adv_cfg.get("enabled", False):
        from adversarial_learning import AdversarialLearner
        from dataset_loader import load_dataset
        from marble_neuronenblitz import Neuronenblitz

        examples = []
        if dataset_path:
            try:
                examples = load_dataset(dataset_path)
            except Exception:  # pragma: no cover - best effort loading
                examples = []
        discriminator = Neuronenblitz(marble.core)
        learner = AdversarialLearner(
            marble.core,
            marble.neuronenblitz,
            discriminator,
            noise_dim=adv_cfg.get("noise_dim", 1),
        )
        if examples:
            real_vals = [float(t) for _, t in examples]
            learner.train(
                real_vals,
                epochs=int(adv_cfg.get("epochs", 1)),
                batch_size=int(adv_cfg.get("batch_size", 1)),
            )
        marble.adversarial_learner = learner

    tl_cfg = cfg.get("transfer_learning", {})
    if tl_cfg.get("enabled", False):
        from dataset_loader import load_dataset
        from transfer_learning import TransferLearner

        examples = []
        if dataset_path:
            try:
                examples = load_dataset(dataset_path)
            except Exception:  # pragma: no cover - best effort loading
                examples = []
        learner = TransferLearner(
            marble.core,
            marble.neuronenblitz,
            freeze_fraction=tl_cfg.get("freeze_fraction", 0.5),
        )
        if examples:
            ex_pairs = [(float(i), float(t)) for i, t in examples]
            learner.train(
                ex_pairs,
                epochs=int(tl_cfg.get("epochs", 1)),
                batch_size=int(tl_cfg.get("batch_size", 1)),
            )
        marble.transfer_learner = learner

    ss_cfg = cfg.get("semi_supervised_learning", {})
    if ss_cfg.get("enabled", False):
        from dataset_loader import load_dataset
        from semi_supervised_learning import SemiSupervisedLearner

        examples = []
        if dataset_path:
            try:
                examples = load_dataset(dataset_path)
            except Exception:  # pragma: no cover - best effort loading
                examples = []
        learner = SemiSupervisedLearner(
            marble.core,
            marble.neuronenblitz,
            unlabeled_weight=ss_cfg.get("unlabeled_weight", 0.5),
        )
        if examples:
            learner.train(
                examples,
                epochs=int(ss_cfg.get("epochs", 1)),
                batch_size=int(ss_cfg.get("batch_size", 1)),
            )
        marble.semi_supervised_learner = learner

    qf_cfg = cfg.get("quantum_flux_learning", {})
    if qf_cfg.get("enabled", False):
        from dataset_loader import load_dataset
        from quantum_flux_learning import QuantumFluxLearner

        examples = []
        if dataset_path:
            try:
                examples = load_dataset(dataset_path)
            except Exception:  # pragma: no cover - best effort loading
                examples = []
        learner = QuantumFluxLearner(
            marble.core,
            marble.neuronenblitz,
            phase_rate=qf_cfg.get("phase_rate", 0.1),
        )
        if examples:
            learner.train(examples, epochs=int(qf_cfg.get("epochs", 1)))
        marble.quantum_flux_learner = learner

    se_cfg = cfg.get("synaptic_echo_learning", {})
    if se_cfg.get("enabled", False):
        from dataset_loader import load_dataset
        from synaptic_echo_learning import SynapticEchoLearner

        examples = []
        if dataset_path:
            try:
                examples = load_dataset(dataset_path)
            except Exception:  # pragma: no cover - best effort loading
                examples = []
        learner = SynapticEchoLearner(
            marble.core,
            marble.neuronenblitz,
            echo_influence=se_cfg.get("echo_influence", 1.0),
        )
        if examples:
            learner.train(examples, epochs=int(se_cfg.get("epochs", 1)))
        marble.synaptic_echo_learner = learner

    fd_cfg = cfg.get("fractal_dimension_learning", {})
    if fd_cfg.get("enabled", False):
        from dataset_loader import load_dataset
        from fractal_dimension_learning import FractalDimensionLearner

        examples = []
        if dataset_path:
            try:
                examples = load_dataset(dataset_path)
            except Exception:  # pragma: no cover - best effort loading
                examples = []
        learner = FractalDimensionLearner(
            marble.core,
            marble.neuronenblitz,
            target_dimension=fd_cfg.get("target_dimension", 4.0),
        )
        if examples:
            learner.train(examples, epochs=int(fd_cfg.get("epochs", 1)))
        marble.fractal_dimension_learner = learner

    return marble
