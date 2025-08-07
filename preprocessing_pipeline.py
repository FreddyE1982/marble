import os
import pickle
import hashlib
from typing import Callable, Iterable, Sequence, Any, List

from remote_worker_pool import RemoteWorkerPool

from tokenizers import Tokenizer
from marble import DataLoader


class PreprocessingPipeline:
    """Apply preprocessing functions to data with result caching."""

    def __init__(
        self,
        steps: Sequence[Callable[[Any], Any]],
        cache_dir: str = "preproc_cache",
        *,
        dataloader: DataLoader | None = None,
        tokenizer: Tokenizer | None = None,
        worker_pool: RemoteWorkerPool | None = None,
    ) -> None:
        self.steps = list(steps)
        self.cache_dir = cache_dir
        if dataloader is not None:
            self.dataloader = dataloader
            if tokenizer is not None:
                self.dataloader.tokenizer = tokenizer
        else:
            self.dataloader = (
                DataLoader(tokenizer=tokenizer) if tokenizer is not None else None
            )
        if worker_pool is None:
            from config_loader import load_config

            cfg = load_config().get("preprocessing", {})
            workers = int(cfg.get("workers", 0))
            worker_pool = RemoteWorkerPool(workers) if workers > 0 else None
        self.worker_pool = worker_pool

    def _cache_path(self, dataset_id: str) -> str:
        os.makedirs(self.cache_dir, exist_ok=True)
        hashed = hashlib.md5(dataset_id.encode("utf-8")).hexdigest()
        return os.path.join(self.cache_dir, f"{hashed}.pkl")

    def process(self, data: Iterable[Any], dataset_id: str) -> List[Any]:
        """Process ``data`` and return the cached result if available."""
        path = self._cache_path(dataset_id)
        if os.path.exists(path):
            with open(path, "rb") as f:
                cached = pickle.load(f)
            if self.dataloader is not None:
                cached = [self.dataloader.decode(item) for item in cached]
            return cached
        processed = list(data)
        for step in self.steps:
            if self.worker_pool is not None:
                processed = self.worker_pool.map(step, processed)
            else:
                processed = [step(item) for item in processed]
        to_store = processed
        if self.dataloader is not None:
            to_store = [self.dataloader.encode(item) for item in processed]
        with open(path, "wb") as f:
            pickle.dump(to_store, f)
        return processed
