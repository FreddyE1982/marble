from __future__ import annotations

import pickle
from typing import Any, Iterable, Tuple, Callable

from tokenizers import Tokenizer
from torch.utils.data import Dataset

from bit_tensor_dataset import BitTensorDataset
from marble_core import DataLoader, Core
from marble_neuronenblitz import Neuronenblitz
from curriculum_learning import CurriculumLearner
from marble_imports import cp


class CurriculumPairsPipeline:
    """Train ``CurriculumLearner`` on ``(input, target)`` pairs."""

    def __init__(
        self,
        core: Core,
        save_path: str = "curriculum.pkl",
        *,
        difficulty_fn: Callable[[tuple[Any, Any]], float] | None = None,
        schedule: str = "linear",
        dataloader: DataLoader | None = None,
        tokenizer: Tokenizer | None = None,
        use_vocab: bool = False,
    ) -> None:
        self.core = core
        if dataloader is not None:
            self.loader = dataloader
            if tokenizer is not None:
                self.loader.tokenizer = tokenizer
        else:
            self.loader = DataLoader(tokenizer=tokenizer)
        self.save_path = save_path
        self.use_vocab = use_vocab
        self.last_dataset: BitTensorDataset | None = None
        self.nb = Neuronenblitz(self.core)
        self.learner = CurriculumLearner(
            self.core, self.nb, difficulty_fn=difficulty_fn, schedule=schedule
        )

    def _to_float(self, obj: Any) -> float:
        tensor = self.loader.encode(obj)
        if hasattr(tensor, "mean"):
            return float(cp.asnumpy(tensor).astype(float).mean())
        return float(tensor)

    def train(self, pairs: Iterable[Tuple[Any, Any]] | Dataset, epochs: int = 1) -> str:
        if isinstance(pairs, Dataset):
            if isinstance(pairs, BitTensorDataset):
                bit_ds = pairs
            else:
                bit_ds = BitTensorDataset([(i, t) for i, t in pairs], use_vocab=self.use_vocab)
        else:
            bit_ds = BitTensorDataset(list(pairs), use_vocab=self.use_vocab)

        self.last_dataset = bit_ds

        examples = [
            (self._to_float(bit_ds.tensor_to_object(inp)), self._to_float(bit_ds.tensor_to_object(tgt)))
            for inp, tgt in bit_ds
        ]

        self.learner.train(examples, epochs=epochs)
        with open(self.save_path, "wb") as f:
            pickle.dump({"core": self.core, "neuronenblitz": self.nb}, f)
        return self.save_path
