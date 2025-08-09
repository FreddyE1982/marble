import re
from pathlib import Path
from typing import List

CUDA_PATTERNS = [
    re.compile(r"torch\.cuda"),
    re.compile(r"\.to\(\s*['\"]cuda['\"]"),
    re.compile(r"\.cuda\("),
]

def scan_repo(root: Path) -> List[str]:
    results: List[str] = []
    for path in root.rglob("*.py"):
        try:
            text = path.read_text(encoding="utf-8")
        except Exception:
            continue
        if any(p.search(text) for p in CUDA_PATTERNS):
            results.append(str(path.relative_to(root)))
    return sorted(results)


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[1]
    for filepath in scan_repo(repo_root):
        print(filepath)
