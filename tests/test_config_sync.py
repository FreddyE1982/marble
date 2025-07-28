import os
from tempfile import TemporaryDirectory
from config_sync_service import sync_config


def test_sync_config(tmp_path):
    src = tmp_path / 'src.yaml'
    dest = tmp_path / 'dest.yaml'
    src.write_text('a: 1')
    sync_config(str(src), [str(dest)])
    assert dest.read_text() == 'a: 1'
