from __future__ import annotations

import pickle
from typing import Any, Iterable, Tuple

from tokenizers import Tokenizer
from torch.utils.data import Dataset

from bit_tensor_dataset import BitTensorDataset
from continual_learning import ReplayContinualLearner
from marble_core import Core, DataLoader
from marble_imports import cp
from marble_neuronenblitz import Neuronenblitz
from marble_brain import Brain


class ContinualPairsPipeline:
    """Train a MARBLE system using :class:`ReplayContinualLearner`."""

    def __init__(
        self,
        core: Core,
        save_path: str = "trained_continual.pkl",
        *,
        dataloader: DataLoader | None = None,
        tokenizer: Tokenizer | None = None,
        memory_size: int = 10,
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
        self.learner = ReplayContinualLearner(self.core, self.nb, memory_size=int(memory_size))

    def _to_float(self, obj: Any) -> float:
        tensor = self.loader.encode(obj)
        if hasattr(tensor, "mean"):
            return float(cp.asnumpy(tensor).astype(float).mean())
        return float(tensor)

    def train(self, pairs: Iterable[Tuple[Any, Any]] | Dataset, epochs: int = 1) -> str:
        if isinstance(pairs, Dataset):
            if isinstance(pairs, BitTensorDataset):
                iter_pairs = (
                    (pairs.tensor_to_object(inp), pairs.tensor_to_object(tgt))
                    for inp, tgt in pairs
                )
            else:
                iter_pairs = ((inp, tgt) for inp, tgt in pairs)
        else:
            iter_pairs = pairs

        examples = [
            (self._to_float(i), self._to_float(t)) for i, t in iter_pairs
        ]
        self.learner.train(examples, epochs=int(epochs))
        with open(self.save_path, "wb") as f:
            pickle.dump({"core": self.core, "neuronenblitz": self.nb}, f)
        return self.save_path
