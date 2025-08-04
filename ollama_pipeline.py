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
4. Provide an interactive console allowing users to converse with the model
   while continually fine‑tuning it on the conversation history.  Training
   pairs are constructed from the last ``history_limit`` messages (user and
   assistant) ensuring that the model learns from recent context.

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


def _format_history(history: List[dict[str, str]]) -> str:
    """Return ``history`` formatted as ``"role: content"`` lines."""

    return "\n".join(f"{m['role']}: {m['content']}" for m in history)


class OllamaInteractiveTrainingPlugin(PipelinePlugin):
    """Pipeline plugin that fine-tunes an Ollama model based on user pairs."""

    def __init__(
        self,
        model: str,
        interactions: Iterable[Tuple[str, str]] | None = None,
        epochs: int = 1,
        history_limit: int = 10,
    ) -> None:
        interactions = [] if interactions is None else list(interactions)
        super().__init__(
            model=model,
            interactions=interactions,
            epochs=int(epochs),
            history_limit=int(history_limit),
        )
        self.model_name = model
        self.interactions = interactions
        self.epochs = int(epochs)
        self.history_limit = int(history_limit)
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

        if self.interactions:
            pipeline.train(self.interactions, epochs=self.epochs)
            for prompt, _expected in self.interactions:
                resp, self.history = chat_with_history(
                    self.core,
                    self.model_name,
                    prompt,
                    self.history,
                    history_limit=self.history_limit,
                )

        outputs = []
        while True:
            try:
                user_message = input("user> ")
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if user_message.strip().lower() in {"quit", "exit"}:
                break

            resp, self.history = chat_with_history(
                self.core,
                self.model_name,
                user_message,
                self.history,
                history_limit=self.history_limit,
            )
            assistant_msg = resp.get("message", {}).get("content", "")
            print(f"assistant> {assistant_msg}")
            context_history = self.history[:-1][-self.history_limit:]
            context_text = _format_history(context_history)
            pipeline.train([(context_text, assistant_msg)], epochs=self.epochs)
            outputs.append(resp)

        return {"core": self.core, "responses": outputs, "history": self.history}


register_plugin("ollama_interactive_train", OllamaInteractiveTrainingPlugin)
