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
4. Train the core on user supplied ``(prompt, response)`` pairs using the
   :class:`UnifiedPairsPipeline` and finally generate replies for the provided
   prompts via :func:`ollama_interop.chat_with_history`.

The plugin automatically utilises GPU acceleration when ``cuda`` is
available, falling back to CPU otherwise.  All heavyweight external
operations (model loading, conversion and API calls) are organised inside the
``initialise`` and ``execute`` lifecycle methods to integrate seamlessly with
the generic :class:`pipeline.Pipeline` infrastructure.
"""

from typing import Iterable, List, Tuple

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
    """Pipeline plugin that fine-tunes an Ollama model based on user pairs."""

    def __init__(
        self,
        model: str,
        interactions: Iterable[Tuple[str, str]],
        epochs: int = 1,
    ) -> None:
        super().__init__(model=model, interactions=list(interactions), epochs=int(epochs))
        self.model_name = model
        self.interactions = list(interactions)
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
        # Train the core on the provided (prompt, response) interactions.
        pipeline = UnifiedPairsPipeline(
            self.core,
            {"autoencoder": AutoencoderLearner(self.core, self.nb)},
            tokenizer=None,
            use_vocab=False,
        )
        pipeline.train(self.interactions, epochs=self.epochs)
        # After training, generate responses for the prompts to verify that the
        # model has been registered correctly with Ollama.
        outputs = []
        for prompt, _expected in self.interactions:
            resp, self.history = chat_with_history(
                self.core, self.model_name, prompt, self.history, history_limit=10
            )
            outputs.append(resp)
        return {"core": self.core, "responses": outputs, "history": self.history}


register_plugin("ollama_interactive_train", OllamaInteractiveTrainingPlugin)
