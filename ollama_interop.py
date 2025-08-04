"""Ollama integration utilities for MARBLE models.

This module allows MARBLE ``Core`` objects to be registered with an
Ollama server directly without converting them to another format.  The
``Core`` is serialised to JSON and embedded in a `Modelfile` which is
then sent to the Ollama API.  Once registered, prompts can be executed
via ``ollama.generate``.

The functions defined here are thin wrappers around the official
``ollama`` Python client and therefore work with both CPU and GPU based
Ollama installations.
"""

from __future__ import annotations

import textwrap
from typing import Any, Dict

from marble_core import Core
from marble_utils import core_to_json

try:  # pragma: no cover - exercised in tests via patching
    import ollama
except Exception as exc:  # pragma: no cover - handled gracefully
    ollama = None
    _IMPORT_ERROR = exc
else:  # pragma: no cover - executed when import succeeds
    _IMPORT_ERROR = None


def _require_ollama() -> None:
    """Ensure the ``ollama`` package is available.

    Raises
    ------
    ImportError
        If the ``ollama`` client could not be imported.  The original
        import error is chained to provide additional context.
    """

    if ollama is None:  # pragma: no cover - triggered only if missing
        raise ImportError("ollama package is required") from _IMPORT_ERROR


def core_to_modelfile(core: Core, model: str) -> str:
    """Return a Modelfile string that embeds ``core``.

    Parameters
    ----------
    core:
        The MARBLE ``Core`` to serialise.
    model:
        Name under which the model will be registered in Ollama.  This is
        included in the Modelfile for documentation purposes only.

    Returns
    -------
    str
        A Modelfile definition that can be passed to ``ollama.create``.
    """

    core_json = core_to_json(core)
    modelfile = f"""
    # Modelfile for MARBLE model: {model}
    FROM embed
    PARAMETER format marble
    WEIGHTS <<'EOF'
    {core_json}
    EOF
    """
    return textwrap.dedent(modelfile)


def register_core(core: Core, model: str, config: Dict[str, Any] | None = None) -> None:
    """Register ``core`` with an Ollama server.

    The function sends the Modelfile produced by :func:`core_to_modelfile`
    to ``ollama.create``.

    Parameters
    ----------
    core:
        MARBLE ``Core`` to register.
    model:
        Name of the model to create in Ollama.
    config:
        Optional configuration dictionary forwarded to ``ollama.create``.
    """

    _require_ollama()
    if config is None:
        config = {}
    modelfile = core_to_modelfile(core, model)
    ollama.create(model=model, modelfile=modelfile, config=config)


def generate(
    core: Core,
    prompt: str,
    model: str,
    config: Dict[str, Any] | None = None,
    **options: Any,
) -> Dict[str, Any]:
    """Generate a response from ``core`` via Ollama.

    If the model has not been previously registered this function will
    register it automatically using :func:`register_core`.

    Parameters
    ----------
    core:
        MARBLE ``Core`` used for inference.
    prompt:
        The textual prompt to send to the model.
    model:
        Name under which the model is (or will be) registered in Ollama.
    config:
        Optional configuration dictionary forwarded to ``register_core``.
    **options:
        Additional keyword arguments forwarded to ``ollama.generate``.

    Returns
    -------
    dict
        The dictionary returned by ``ollama.generate``.
    """

    _require_ollama()
    register_core(core, model, config)
    return ollama.generate(model=model, prompt=prompt, **options)
