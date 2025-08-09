from pathlib import Path
import subprocess
from scan_cuda_modules import scan_repo


def find_modules_missing_cpu_tests(repo_root: Path):
    cuda_modules = [m for m in scan_repo(repo_root) if not m.startswith("tests/")]
    tests_dir = repo_root / "tests"
    missing = []
    for module in cuda_modules:
        mod_stem = Path(module).stem
        # search tests directory for module name
        result = subprocess.run(
            ["rg", mod_stem, str(tests_dir)],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0 or not result.stdout.strip():
            missing.append(module)
    return missing


def generate_report(repo_root: Path, missing):
    report_path = repo_root / "cpu_fallback_report.md"
    lines = ["# Modules lacking CPU fallback tests", ""]
    if missing:
        for m in missing:
            lines.append(f"- `{m}`")
    else:
        lines.append("All modules with CUDA usage have associated tests.")
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    missing = find_modules_missing_cpu_tests(root)
    report = generate_report(root, missing)
    for m in missing:
        print(m)
    print(f"Report written to {report}")
