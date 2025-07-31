from __future__ import annotations

import pickle
from typing import Any, Iterable

from tokenizers import Tokenizer
from torch.utils.data import Dataset

from bit_tensor_dataset import BitTensorDataset
from marble_core import DataLoader, Core
from marble_neuronenblitz import Neuronenblitz
from quantum_flux_learning import QuantumFluxLearner
from marble_imports import cp


class QuantumFluxPairsPipeline:
    """Train ``QuantumFluxLearner`` on ``(input, target)`` pairs."""

    def __init__(
        self,
        core: Core,
        save_path: str = "quantum_flux.pkl",
        *,
        phase_rate: float = 0.1,
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
        self.learner = QuantumFluxLearner(self.core, self.nb, phase_rate=phase_rate)

    def _to_float(self, obj: Any) -> float:
        tensor = self.loader.encode(obj)
        if hasattr(tensor, "mean"):
            return float(cp.asnumpy(tensor).astype(float).mean())
        return float(tensor)

    def train(self, pairs: Iterable[tuple[Any, Any]] | Dataset, epochs: int = 1) -> str:
        if isinstance(pairs, Dataset):
            if isinstance(pairs, BitTensorDataset):
                bit_ds = pairs
            else:
                bit_ds = BitTensorDataset([(i, t) for i, t in pairs], use_vocab=self.use_vocab)
        else:
            bit_ds = BitTensorDataset(list(pairs), use_vocab=self.use_vocab)

        self.last_dataset = bit_ds
        list_pairs = [
            (self._to_float(bit_ds.tensor_to_object(inp)), self._to_float(bit_ds.tensor_to_object(tgt)))
            for inp, tgt in bit_ds
        ]
        self.learner.train(list_pairs, epochs=epochs)
        with open(self.save_path, "wb") as f:
            pickle.dump({"core": self.core, "neuronenblitz": self.nb}, f)
        return self.save_path
