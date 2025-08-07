import pathlib
import sys

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
from find_unimplemented_params import find_unimplemented


def test_find_unimplemented_params(tmp_path):
    md = tmp_path / "CONFIGURABLE_PARAMETERS.md"
    md.write_text("## section\n- present\n- missing\n")
    (tmp_path / "code.py").write_text("present=1\n")
    missing = find_unimplemented(md, tmp_path)
    assert missing == ["missing"]
