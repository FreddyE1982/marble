from __future__ import annotations

from typing import Any, Iterable, Tuple
import pickle

from diffusion_core import DiffusionCore
from marble_neuronenblitz import Neuronenblitz
from marble_brain import Brain
from marble_imports import cp
from marble_core import DataLoader
from tokenizers import Tokenizer


class DiffusionPairsPipeline:
    """Train a MARBLE with ``DiffusionCore`` on ``(input, expected)`` pairs.

    The pipeline converts arbitrary input and target objects to numeric values
    using the :class:`DataLoader` embedded in ``DiffusionCore`` so it works with
    any data type. After training for the requested number of epochs, the full
    model (core and neuronenblitz) is pickled to ``save_path`` for later
    inference.
    """

    def __init__(
        self,
        core: DiffusionCore,
        save_path: str = "trained_marble.pkl",
        *,
        dataloader: DataLoader | None = None,
        tokenizer: Tokenizer | None = None,
    ) -> None:
        self.core = core
        if dataloader is not None:
            self.loader = dataloader
            if tokenizer is not None:
                self.loader.tokenizer = tokenizer
        elif hasattr(core, "loader"):
            if tokenizer is not None:
                core.loader.tokenizer = tokenizer
            self.loader = core.loader
        else:
            self.loader = DataLoader(tokenizer=tokenizer)
        self.save_path = save_path
        self.nb = Neuronenblitz(self.core)
        self.brain = Brain(self.core, self.nb, self.loader)

    def _to_float(self, obj: Any) -> float:
        tensor = self.loader.encode(obj)
        if hasattr(tensor, "mean"):
            return float(cp.asnumpy(tensor).astype(float).mean())
        return float(tensor)

    def train(self, pairs: Iterable[Tuple[Any, Any]], epochs: int = 1) -> str:
        examples = [(self._to_float(i), self._to_float(t)) for i, t in pairs]
        self.nb.train(examples, epochs=epochs)
        with open(self.save_path, "wb") as f:
            pickle.dump({"core": self.core, "neuronenblitz": self.nb}, f)
        return self.save_path
