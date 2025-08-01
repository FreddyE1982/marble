from dataset_versioning import create_version, apply_version, revert_version


def test_create_and_apply_version(tmp_path):
    base = [(1, 2), (3, 4)]
    new = [(1, 2), (5, 6)]
    vid = create_version(base, new, tmp_path)
    patched = apply_version(base, tmp_path, vid)
    assert patched == new
    reverted = revert_version(patched, tmp_path, vid)
    assert reverted == base
