import os
import json
import requests
from typing import Sequence
from tqdm import tqdm


def replicate_dataset(path: str, urls: Sequence[str]) -> None:
    """Send dataset at ``path`` to all ``urls`` with progress output."""
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path, "rb") as f:
        data = f.read()
    for url in urls:
        resp = requests.post(
            url.rstrip("/") + "/replicate",
            data=data,
            headers={"Content-Type": "application/octet-stream"},
            stream=True,
        )
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        for _ in tqdm(resp.iter_content(chunk_size=8192), total=total, desc=f"send->{url}"):
            pass
