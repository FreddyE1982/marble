from __future__ import annotations
import os
import subprocess
import sys
from pathlib import Path


def discover_test_files(root: Path) -> list[Path]:
    """Collect all pytest files under the tests directory."""
    tests_dir = root / "tests"
    return sorted(p for p in tests_dir.rglob("test_*.py"))


def run_tests_cpu_only(test_files: list[Path]) -> int:
    """Run each test file with CUDA disabled. Returns number of failures."""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    failures: list[str] = []
    for test in test_files:
        print(f"Running {test} with CUDA disabled", flush=True)
        result = subprocess.run(
            [sys.executable, "-m", "pytest", str(test)], env=env
        )
        if result.returncode != 0:
            failures.append(str(test))
    if failures:
        print("Failed tests:")
        for f in failures:
            print(f" - {f}")
    else:
        print("All tests passed without CUDA.")
    return len(failures)


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    tests = discover_test_files(repo_root)
    failures = run_tests_cpu_only(tests)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
