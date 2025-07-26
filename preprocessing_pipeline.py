import os
import pickle
import hashlib
from typing import Callable, Iterable, Sequence, Any, List


class PreprocessingPipeline:
    """Apply preprocessing functions to data with result caching."""

    def __init__(self, steps: Sequence[Callable[[Any], Any]], cache_dir: str = "preproc_cache") -> None:
        self.steps = list(steps)
        self.cache_dir = cache_dir

    def _cache_path(self, dataset_id: str) -> str:
        os.makedirs(self.cache_dir, exist_ok=True)
        hashed = hashlib.md5(dataset_id.encode("utf-8")).hexdigest()
        return os.path.join(self.cache_dir, f"{hashed}.pkl")

    def process(self, data: Iterable[Any], dataset_id: str) -> List[Any]:
        """Process ``data`` and return the cached result if available."""
        path = self._cache_path(dataset_id)
        if os.path.exists(path):
            with open(path, "rb") as f:
                return pickle.load(f)
        processed = list(data)
        for step in self.steps:
            processed = [step(item) for item in processed]
        with open(path, "wb") as f:
            pickle.dump(processed, f)
        return processed
