from pathlib import Path
import yaml

from marble_main import MARBLE
from neuromodulatory_system import NeuromodulatorySystem
from meta_parameter_controller import MetaParameterController
from marble_core import MemorySystem
from remote_offload import RemoteBrainClient

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
    long_term_path = cfg.get("memory_system", {}).get(
        "long_term_path", "long_term_memory.pkl"
    )
    memory_system = MemorySystem(long_term_path)

    brain_params.update({
        "neuromodulatory_system": neuromod_system,
        "meta_controller": meta_controller,
        "memory_system": memory_system,
    })

    remote_client = None
    remote_cfg = cfg.get("remote_client", {})
    if isinstance(remote_cfg, dict) and remote_cfg.get("url"):
        remote_client = RemoteBrainClient(remote_cfg["url"])

    marble = MARBLE(
        core_params,
        formula=formula,
        formula_num_neurons=formula_num_neurons,
        nb_params=nb_params,
        brain_params=brain_params,
        remote_client=remote_client,
    )
    return marble
