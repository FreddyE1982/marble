import yaml

from config_loader import create_marble_from_config, load_config


def test_sync_service_from_config(tmp_path):
    cfg = load_config()
    cfg["sync"] = {"interval_ms": 5}
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))

    marble = create_marble_from_config(str(cfg_path))
    assert hasattr(marble, "tensor_sync_service")
    assert abs(marble.tensor_sync_service.interval - 0.005) < 1e-9
