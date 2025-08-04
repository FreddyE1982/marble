from __future__ import annotations

"""Pipeline plugin enabling interactive training with Ollama models.

The module defines :class:`OllamaInteractiveTrainingPlugin` which performs
four high level steps:

1. Ensure the requested model is available locally by invoking
   :func:`ollama.pull`.
2. Load the model with the Hugging Face :mod:`transformers` library and
   convert it into a MARBLE :class:`~marble_core.Core` via
   :func:`pytorch_to_marble.convert_model`.
3. Register the resulting core with an Ollama server so that prompts can be
   served through the standard ``ollama`` API.
4. Interact with the model through :func:`ollama_interop.chat_with_history`
   while continuously fine‑tuning on the recent conversation history via the
   :class:`UnifiedPairsPipeline`.

The plugin automatically utilises GPU acceleration when ``cuda`` is
available, falling back to CPU otherwise.  All heavyweight external
operations (model loading, conversion and API calls) are organised inside the
``initialise`` and ``execute`` lifecycle methods to integrate seamlessly with
the generic :class:`pipeline.Pipeline` infrastructure.
"""

from typing import Iterable, List

import torch

import ollama
from transformers import AutoModelForCausalLM

from marble_core import Core
from marble_neuronenblitz import Neuronenblitz
from ollama_interop import chat_with_history, register_core
from pipeline_plugins import PipelinePlugin, register_plugin
from pytorch_to_marble import convert_model
from unified_pairs_pipeline import UnifiedPairsPipeline
from autoencoder_learning import AutoencoderLearner


class OllamaInteractiveTrainingPlugin(PipelinePlugin):
    """Pipeline plugin that fine-tunes an Ollama model during conversation."""

    def __init__(
        self,
        model: str,
        prompts: Iterable[str],
        history_limit: int = 10,
        epochs: int = 1,
    ) -> None:
        super().__init__(
            model=model,
            prompts=list(prompts),
            history_limit=int(history_limit),
            epochs=int(epochs),
        )
        self.model_name = model
        self.prompts = list(prompts)
        self.history_limit = int(history_limit)
        self.epochs = int(epochs)
        self.history: List[dict[str, str]] = []

    def initialise(self, device: torch.device, marble: Core | None = None) -> None:
        self.device = device
        # Ensure model is present locally; ``ollama.pull`` is a no-op when the
        # model already exists.  It transparently downloads either CPU or GPU
        # weights depending on the server configuration.
        ollama.pull(self.model_name)
        hf_model = AutoModelForCausalLM.from_pretrained(self.model_name)
        hf_model.to(device)
        self.core = convert_model(hf_model)
        self.nb = Neuronenblitz(self.core)
        register_core(self.core, self.model_name)

    def execute(self, device: torch.device, marble: Core | None = None):
        pipeline = UnifiedPairsPipeline(
            self.core,
            {"autoencoder": AutoencoderLearner(self.core, self.nb)},
            tokenizer=None,
            use_vocab=False,
        )
        outputs = []
        for prompt in self.prompts:
            resp, self.history = chat_with_history(
                self.core, self.model_name, prompt, self.history, self.history_limit
            )
            outputs.append(resp)
            truncated = self.history[-self.history_limit :]
            pairs = [
                (truncated[i]["content"], truncated[i + 1]["content"])
                for i in range(len(truncated) - 1)
                if truncated[i]["role"] == "user"
                and truncated[i + 1]["role"] == "assistant"
            ]
            if pairs:
                pipeline.train(pairs, epochs=self.epochs)
        return {"core": self.core, "responses": outputs, "history": self.history}


register_plugin("ollama_interactive_train", OllamaInteractiveTrainingPlugin)
