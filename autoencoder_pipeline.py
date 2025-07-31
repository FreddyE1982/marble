from __future__ import annotations

import pickle
from typing import Any, Iterable

from tokenizers import Tokenizer
from torch.utils.data import Dataset

from bit_tensor_dataset import BitTensorDataset
from marble_core import DataLoader, Core
from marble_neuronenblitz import Neuronenblitz
from autoencoder_learning import AutoencoderLearner
from marble_imports import cp


class AutoencoderPipeline:
    """Train an :class:`AutoencoderLearner` on arbitrary objects."""

    def __init__(
        self,
        core: Core,
        save_path: str = "autoencoder.pkl",
        *,
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
        self.learner = AutoencoderLearner(self.core, self.nb)

    def _to_float(self, obj: Any) -> float:
        tensor = self.loader.encode(obj)
        if hasattr(tensor, "mean"):
            return float(cp.asnumpy(tensor).astype(float).mean())
        return float(tensor)

    def train(self, data: Iterable[Any] | Dataset, epochs: int = 1) -> str:
        if isinstance(data, Dataset):
            if isinstance(data, BitTensorDataset):
                bit_ds = data
            else:
                bit_ds = BitTensorDataset([(d, d) for d in data], use_vocab=self.use_vocab)
        else:
            bit_ds = BitTensorDataset([(d, d) for d in list(data)], use_vocab=self.use_vocab)

        iter_values = (bit_ds.tensor_to_object(inp) for inp, _ in bit_ds)
        values = [self._to_float(v) for v in iter_values]
        self.last_dataset = bit_ds
        self.learner.train(values, epochs=epochs)
        with open(self.save_path, "wb") as f:
            pickle.dump({"core": self.core, "neuronenblitz": self.nb}, f)
        return self.save_path
