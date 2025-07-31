from __future__ import annotations

import pickle
from typing import Any, Iterable

from tokenizers import Tokenizer
from torch.utils.data import Dataset

from bit_tensor_dataset import BitTensorDataset
from autoencoder_learning import AutoencoderLearner
from marble_core import Core, DataLoader
from marble_imports import cp
from marble_neuronenblitz import Neuronenblitz
from marble_brain import Brain


class AutoencoderPairsPipeline:
    """Train a MARBLE autoencoder using :class:`AutoencoderLearner`."""

    def __init__(
        self,
        core: Core,
        save_path: str = "trained_autoencoder.pkl",
        *,
        dataloader: DataLoader | None = None,
        tokenizer: Tokenizer | None = None,
        noise_std: float = 0.1,
        noise_decay: float = 0.99,
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
        self.learner = AutoencoderLearner(
            self.core, self.nb, noise_std=float(noise_std), noise_decay=float(noise_decay)
        )

    def _to_float(self, obj: Any) -> float:
        tensor = self.loader.encode(obj)
        if hasattr(tensor, "mean"):
            return float(cp.asnumpy(tensor).astype(float).mean())
        return float(tensor)

    def train(self, data: Iterable[Any] | Dataset, epochs: int = 1) -> str:
        if isinstance(data, Dataset):
            if isinstance(data, BitTensorDataset):
                iter_inputs = (data.tensor_to_object(inp) for inp, _ in data)
            else:
                iter_inputs = (
                    inp if not isinstance(inp, tuple) else inp[0] for inp in data
                )
        else:
            iter_inputs = data

        values = [self._to_float(i) for i in iter_inputs]
        self.learner.train(values, epochs=int(epochs))
        with open(self.save_path, "wb") as f:
            pickle.dump({"core": self.core, "neuronenblitz": self.nb}, f)
        return self.save_path
