import time
from tempfile import TemporaryDirectory
from config_sync_service import ConfigSyncService


def test_config_sync_service(tmp_path):
    src = tmp_path / "src.yaml"
    dst1 = tmp_path / "node1" / "cfg.yaml"
    dst2 = tmp_path / "node2" / "cfg.yaml"
    src.write_text("a: 1")
    svc = ConfigSyncService(str(src), [str(dst1), str(dst2)])
    svc.start()
    try:
        src.write_text("a: 2")
        time.sleep(0.2)
        assert dst1.read_text() == "a: 2"
        assert dst2.read_text() == "a: 2"
    finally:
        svc.stop()
