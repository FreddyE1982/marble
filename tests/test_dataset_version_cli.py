from dataset_loader import load_dataset
from dataset_version_cli import create_version_cmd, list_versions, switch_version


def test_dataset_version_cli_cycle(tmp_path):
    base = tmp_path / "base.csv"
    base.write_text("input,target\n1,2\n")
    new = tmp_path / "new.csv"
    new.write_text("input,target\n1,2\n3,4\n")
    registry = tmp_path / "versions"

    vid = create_version_cmd(str(base), str(new), str(registry))
    assert vid in list_versions(str(registry))

    out = tmp_path / "patched.csv"
    switch_version(str(base), vid, str(registry), str(out))
    data = load_dataset(str(out))
    assert len(data) == 2
