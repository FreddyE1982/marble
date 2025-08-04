import os
import sys
from unittest.mock import MagicMock, mock_open, patch

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from marble_core import Core
from tests.test_core_functions import minimal_params

@patch("builtins.input", side_effect=["hi", "quit"])
@patch("ollama_pipeline.open", new_callable=mock_open)
@patch("ollama_pipeline.core_to_json")
@patch("ollama_pipeline.save_marble_system")
@patch("ollama_pipeline.MARBLE")
@patch("ollama_pipeline.chat_with_history")
@patch("ollama_pipeline.UnifiedPairsPipeline")
@patch("ollama_pipeline.Neuronenblitz")
@patch("ollama_pipeline.register_core")
@patch("ollama_pipeline.convert_model")
@patch("ollama_pipeline.AutoModelForCausalLM")
@patch("ollama_pipeline.ollama.pull")
def test_plugin_executes_full_flow(
    pull,
    auto_model,
    convert,
    register,
    nb,
    pairs,
    chat,
    marble_cls,
    save_marble,
    core_to_json,
    m_open,
    mock_input,
):
    pull.return_value = None
    auto_model.from_pretrained.return_value = torch.nn.Linear(1, 1)
    core = Core(minimal_params())
    convert.return_value = core
    nb.return_value = MagicMock()
    pipeline_instance = MagicMock()
    pairs.return_value = pipeline_instance
    pipeline_instance.train.return_value = None
    chat.return_value = (
        {"message": {"content": "ok"}},
        [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "ok"}],
    )
    marble_obj = MagicMock()
    marble_obj.neuronenblitz = MagicMock()
    marble_obj.brain = MagicMock()
    marble_cls.return_value = marble_obj
    core_to_json.return_value = "{}"

    from ollama_pipeline import OllamaInteractiveTrainingPlugin

    plugin = OllamaInteractiveTrainingPlugin("tiny")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    plugin.initialise(device)
    result = plugin.execute(device)

    pull.assert_called_once_with("tiny")
    auto_model.from_pretrained.assert_called_once_with("tiny")
    convert.assert_called_once()
    register.assert_called_once_with(core, "tiny")
    pipeline_instance.train.assert_called_once_with([("user: hi", "ok")], epochs=1)
    chat.assert_called_once_with(core, "tiny", "hi", [], history_limit=10)
    save_marble.assert_called_once()
    core_to_json.assert_called_once_with(core)
    assert m_open.call_count == 1
    assert result["responses"][0]["message"]["content"] == "ok"
