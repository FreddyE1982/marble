from __future__ import annotations

from pathlib import Path
from typing import Optional

from huggingface_hub import login as hf_login, hf_hub_download

HF_TOKEN_PATH = Path.home() / ".huggingface_token"
_logged_in = False

def auto_hf_login(token_path: Optional[str | Path] = None) -> None:
    """Login to Hugging Face if a token file exists and login hasn't happened."""
    global _logged_in
    if _logged_in:
        return
    path = Path(token_path or HF_TOKEN_PATH).expanduser()
    if path.exists():
        token = path.read_text().strip()
        if token:
            hf_login(token=token)
            _logged_in = True

def download_hf_model(repo_id: str, filename: str, *, cache_dir: str | None = None) -> str:
    """Download ``filename`` from ``repo_id`` and return the local path."""
    auto_hf_login()
    return hf_hub_download(repo_id, filename, cache_dir=cache_dir)
