import os
import sys
from unittest.mock import patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from marble_core import Core
from ollama_interop import core_to_modelfile, generate, register_core
from tests.test_core_functions import minimal_params


def test_core_to_modelfile_contains_json():
    params = minimal_params()
    core = Core(params)
    modelfile = core_to_modelfile(core, "mtest")
    assert "PARAMETER format marble" in modelfile
    assert '"neurons"' in modelfile


@patch("ollama_interop.ollama.create")
def test_register_core_invokes_ollama_create(mock_create):
    params = minimal_params()
    core = Core(params)
    register_core(core, "mtest")
    mock_create.assert_called_once()


@patch("ollama_interop.ollama.generate")
@patch("ollama_interop.ollama.create")
def test_generate_registers_and_calls_generate(mock_create, mock_generate):
    params = minimal_params()
    core = Core(params)
    mock_generate.return_value = {"response": "ok"}
    out = generate(core, "hi", "mtest")
    mock_create.assert_called_once()
    mock_generate.assert_called_once()
    assert out["response"] == "ok"
