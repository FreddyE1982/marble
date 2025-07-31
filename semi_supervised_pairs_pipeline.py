from __future__ import annotations

import pickle
from typing import Any, Iterable

from tokenizers import Tokenizer
from torch.utils.data import Dataset

from bit_tensor_dataset import BitTensorDataset
from marble_core import DataLoader, Core
from marble_neuronenblitz import Neuronenblitz
from semi_supervised_learning import SemiSupervisedLearner
from marble_imports import cp


class SemiSupervisedPairsPipeline:
    """Train ``SemiSupervisedLearner`` using labeled and unlabeled data."""

    def __init__(
        self,
        core: Core,
        save_path: str = "semi_supervised.pkl",
        *,
        unlabeled_weight: float = 0.5,
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
        self.labeled_dataset: BitTensorDataset | None = None
        self.unlabeled_dataset: BitTensorDataset | None = None
        self.nb = Neuronenblitz(self.core)
        self.learner = SemiSupervisedLearner(self.core, self.nb, unlabeled_weight=unlabeled_weight)

    def _to_float(self, obj: Any) -> float:
        tensor = self.loader.encode(obj)
        if hasattr(tensor, "mean"):
            return float(cp.asnumpy(tensor).astype(float).mean())
        return float(tensor)

    def train(
        self,
        labeled_pairs: Iterable[tuple[Any, Any]] | Dataset,
        unlabeled_inputs: Iterable[Any] | Dataset,
        epochs: int = 1,
    ) -> str:
        if isinstance(labeled_pairs, Dataset):
            if isinstance(labeled_pairs, BitTensorDataset):
                labeled_ds = labeled_pairs
            else:
                labeled_ds = BitTensorDataset([(i, t) for i, t in labeled_pairs], use_vocab=self.use_vocab)
        else:
            labeled_ds = BitTensorDataset(list(labeled_pairs), use_vocab=self.use_vocab)

        if isinstance(unlabeled_inputs, Dataset):
            if isinstance(unlabeled_inputs, BitTensorDataset):
                unlabeled_ds = unlabeled_inputs
            else:
                unlabeled_ds = BitTensorDataset([(i, i) for i in unlabeled_inputs], use_vocab=self.use_vocab)
        else:
            unlabeled_ds = BitTensorDataset([(i, i) for i in list(unlabeled_inputs)], use_vocab=self.use_vocab)

        self.labeled_dataset = labeled_ds
        self.unlabeled_dataset = unlabeled_ds

        labeled_values = [
            (self._to_float(labeled_ds.tensor_to_object(inp)), self._to_float(labeled_ds.tensor_to_object(tgt)))
            for inp, tgt in labeled_ds
        ]
        unlabeled_values = [self._to_float(unlabeled_ds.tensor_to_object(inp)) for inp, _ in unlabeled_ds]

        self.learner.train(labeled_values, unlabeled_values, epochs=epochs)
        with open(self.save_path, "wb") as f:
            pickle.dump({"core": self.core, "neuronenblitz": self.nb}, f)
        return self.save_path
