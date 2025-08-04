import os
import sys
from unittest.mock import MagicMock, patch

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from marble_core import Core
from tests.test_core_functions import minimal_params


@patch("ollama_pipeline.chat_with_history")
@patch("ollama_pipeline.UnifiedPairsPipeline")
@patch("ollama_pipeline.Neuronenblitz")
@patch("ollama_pipeline.register_core")
@patch("ollama_pipeline.convert_model")
@patch("ollama_pipeline.AutoModelForCausalLM")
@patch("ollama_pipeline.ollama.pull")
def test_plugin_trains_on_recent_history(pull, auto_model, convert, register, nb, pairs, chat):
    pull.return_value = None
    auto_model.from_pretrained.return_value = torch.nn.Linear(1, 1)
    core = Core(minimal_params())
    convert.return_value = core
    nb.return_value = MagicMock()
    pipeline_instance = MagicMock()
    pairs.return_value = pipeline_instance
    pipeline_instance.train.return_value = None
    chat.side_effect = [
        (
            {"message": {"content": "a1"}},
            [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "a1"},
            ],
        ),
        (
            {"message": {"content": "a2"}},
            [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "a1"},
                {"role": "user", "content": "bye"},
                {"role": "assistant", "content": "a2"},
            ],
        ),
    ]

    from ollama_pipeline import OllamaInteractiveTrainingPlugin

    plugin = OllamaInteractiveTrainingPlugin("tiny", ["hi", "bye"], history_limit=4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    plugin.initialise(device)
    result = plugin.execute(device)

    pull.assert_called_once_with("tiny")
    auto_model.from_pretrained.assert_called_once_with("tiny")
    convert.assert_called_once()
    register.assert_called_once_with(core, "tiny")
    assert pipeline_instance.train.call_count == 2
    first_pairs = pipeline_instance.train.call_args_list[0].args[0]
    second_pairs = pipeline_instance.train.call_args_list[1].args[0]
    assert first_pairs == [("hi", "a1")]
    assert second_pairs == [("hi", "a1"), ("bye", "a2")]
    assert chat.call_count == 2
    assert result["responses"][1]["message"]["content"] == "a2"
