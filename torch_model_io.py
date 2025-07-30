import torch
from pathlib import Path
from typing import Any, Dict, Optional

try:
    from safetensors.torch import save_file as st_save_file, load_file as st_load_file
except Exception:  # pragma: no cover - safetensors optional
    st_save_file = None
    st_load_file = None


def save_state_dict(model: torch.nn.Module, path: str) -> None:
    """Save model parameters to ``path`` using :func:`torch.save`."""
    torch.save(model.state_dict(), path)


def load_state_dict(
    model_class: type,
    path: str,
    device: Optional[str | torch.device] = None,
    strict: bool = True,
    **init_kwargs: Any,
) -> torch.nn.Module:
    """Return a model of ``model_class`` loaded from a saved ``state_dict``."""
    device = torch.device(device or "cpu")
    sd = torch.load(path, map_location=device, weights_only=True)
    model = model_class(**init_kwargs)
    model.load_state_dict(sd, strict=strict)
    model.to(device)
    model.eval()
    return model


def save_entire_model(model: torch.nn.Module, path: str) -> None:
    """Serialize ``model`` to ``path``."""
    torch.save(model, path)


def load_entire_model(path: str, device: Optional[str | torch.device] = None) -> torch.nn.Module:
    """Load a model saved with :func:`save_entire_model`."""
    device = torch.device(device or "cpu")
    model = torch.load(path, map_location=device, weights_only=False)
    model.to(device)
    model.eval()
    return model


def save_exported_program(ep: torch.export.ExportedProgram, path: str) -> None:
    """Save ``ExportedProgram`` to ``path`` using :func:`torch.export.save`."""
    torch.export.save(ep, path)


def load_exported_program(path: str) -> torch.export.ExportedProgram:
    """Load ``ExportedProgram`` from ``path``."""
    return torch.export.load(path)


def save_checkpoint(state: Dict[str, Any], path: str) -> None:
    """Persist checkpoint ``state`` to ``path``."""
    torch.save(state, path)


def load_checkpoint(path: str, device: Optional[str | torch.device] = None) -> Dict[str, Any]:
    """Return checkpoint dictionary stored at ``path``."""
    device = torch.device(device or "cpu")
    return torch.load(path, map_location=device, weights_only=True)


def save_multi_model(states: Dict[str, Any], path: str) -> None:
    """Save multiple model and optimizer states in one file."""
    torch.save(states, path)


def load_multi_model(path: str, device: Optional[str | torch.device] = None) -> Dict[str, Any]:
    """Load dictionary produced by :func:`save_multi_model`."""
    device = torch.device(device or "cpu")
    return torch.load(path, map_location=device, weights_only=True)


def warmstart_model(
    model: torch.nn.Module,
    state_dict_path: str,
    device: Optional[str | torch.device] = None,
    strict: bool = False,
) -> torch.nn.Module:
    """Load parameters from ``state_dict_path`` into ``model`` with ``strict=False``."""
    device = torch.device(device or "cpu")
    sd = torch.load(state_dict_path, map_location=device, weights_only=True)
    model.load_state_dict(sd, strict=strict)
    model.to(device)
    return model


# ===== safetensors helpers =====

def save_safetensors(model: torch.nn.Module, path: str) -> None:
    """Save model ``state_dict`` using ``safetensors`` if available."""
    if st_save_file is None:
        raise RuntimeError("safetensors not installed")
    st_save_file(model.state_dict(), path)


def load_safetensors(
    model_class: type,
    path: str,
    device: Optional[str | torch.device] = None,
    strict: bool = True,
    **init_kwargs: Any,
) -> torch.nn.Module:
    """Load a model from a safetensors ``state_dict`` file."""
    if st_load_file is None:
        raise RuntimeError("safetensors not installed")
    device = torch.device(device or "cpu")
    sd = st_load_file(path, device=str(device))
    model = model_class(**init_kwargs)
    model.load_state_dict(sd, strict=strict)
    model.to(device)
    model.eval()
    return model


# ===== convenience autodetection =====

def load_model_auto(
    path: str,
    model_class: Optional[type] = None,
    device: Optional[str | torch.device] = None,
    **kwargs: Any,
) -> torch.nn.Module:
    """Load a model from ``path`` inferring the correct format from the extension."""
    ext = Path(path).suffix
    if ext == ".pt2":
        return load_exported_program(path)
    if ext == ".safetensors":
        if model_class is None:
            raise ValueError("model_class required for safetensors files")
        return load_safetensors(model_class, path, device=device, **kwargs)
    try:
        if model_class is None:
            return load_entire_model(path, device=device)
        return load_state_dict(model_class, path, device=device, **kwargs)
    except Exception:
        return load_entire_model(path, device=device)

