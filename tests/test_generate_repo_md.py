from pathlib import Path
from generate_repo_md import build_repo_markdown


def test_build_repo_markdown(tmp_path: Path) -> None:
    (tmp_path / 'one.md').write_text('md1')
    (tmp_path / 'two.py').write_text('print("hi")')
    # existing repo.md should be overwritten and excluded
    (tmp_path / 'repo.md').write_text('old')
    build_repo_markdown(tmp_path)
    out = (tmp_path / 'repo.md').read_text()
    assert 'old' not in out
    md_index = out.index('# one.md')
    code_index = out.index('# two.py')
    assert md_index < code_index
    assert 'md1' in out
    assert 'print("hi")' in out

