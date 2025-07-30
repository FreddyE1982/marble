import os
from typing import Any, Iterable

from datasets import load_dataset
from huggingface_hub import HfApi, login
from transformers import AutoModel

HF_TOKEN_FILE = os.path.expanduser("~/.cache/marble/hf_token")


def hf_login(token: str | None = None, token_file: str = HF_TOKEN_FILE) -> str | None:
    """Login to the Hugging Face Hub using ``token``.

    If ``token`` is ``None`` and ``token_file`` exists, the token is read from the
    file. After a successful login the token is saved back to ``token_file``. The
    token is returned so callers can pass it to API functions that expect it.
    """
    if token is None:
        token_path = os.path.expanduser(token_file)
        if os.path.exists(token_path):
            with open(token_path, "r", encoding="utf-8") as f:
                token = f.read().strip()
            if token:
                login(token=token)
                return token
        return None

    login(token=token)
    token_path = os.path.expanduser(token_file)
    os.makedirs(os.path.dirname(token_path), exist_ok=True)
    with open(token_path, "w", encoding="utf-8") as f:
        f.write(token.strip())
    return token


def hf_load_dataset(
    dataset_name: str,
    split: str,
    input_key: str = "input",
    target_key: str = "target",
    limit: int | None = None,
    streaming: bool = False,
) -> list[tuple[Any, Any]]:
    """Return ``(input, target)`` pairs from a Hugging Face dataset."""
    token = hf_login()
    ds = load_dataset(dataset_name, split=split, token=token, streaming=streaming)
    examples: list[tuple[Any, Any]] = []
    for record in ds:
        examples.append((record[input_key], record[target_key]))
        if limit is not None and len(examples) >= limit:
            break
    return examples


def hf_load_model(model_name: str):
    """Return a pretrained model from the Hugging Face Hub."""
    hf_login()
    return AutoModel.from_pretrained(model_name, trust_remote_code=True)


def search_hf_datasets(query: str, limit: int = 20) -> list[str]:
    """Return dataset IDs from the Hugging Face Hub matching ``query``."""
    hf_login()
    datasets = HfApi().list_datasets(search=query, limit=limit)
    return [d.id for d in datasets]


def search_hf_models(query: str, limit: int = 20) -> list[str]:
    """Return model IDs from the Hugging Face Hub matching ``query``."""
    hf_login()
    models = HfApi().list_models(search=query, limit=limit)
    return [m.id for m in models]
