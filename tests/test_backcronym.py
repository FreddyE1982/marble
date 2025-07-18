import os
import re


def test_readme_contains_backcronyms_list():
    readme_path = os.path.join(os.path.dirname(__file__), "..", "README.md")
    assert os.path.exists(readme_path)
    with open(readme_path, "r") as f:
        text = f.read()
    # Find lines under the backcronyms section
    match = re.search(r"## Possible MARBLE Backcronyms\n(.*)", text, re.DOTALL)
    assert match, "Backcronyms section missing"
    lines = [line for line in match.group(1).splitlines() if line.startswith("-")]
    assert len(lines) == 10, f"Expected 10 backcronyms, found {len(lines)}"
    assert "Mandelbrot Adaptive Reasoning Brain-Like Engine" in lines[0]
