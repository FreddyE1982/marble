import yaml
from marble_registry import MarbleRegistry
from tests.test_core_functions import minimal_params


def create_cfg(tmp_path):
    cfg = {"core": minimal_params(), "brain": {"save_dir": str(tmp_path)}}
    path = tmp_path / "cfg.yaml"
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f)
    return str(path)


def test_registry_create_get_delete(tmp_path):
    cfg = create_cfg(tmp_path)
    reg = MarbleRegistry()
    reg.create("A", cfg_path=cfg)
    assert reg.list() == ["A"]
    assert reg.get("A") is not None
    reg.delete("A")
    assert reg.list() == []


def test_registry_duplicate(tmp_path):
    cfg = create_cfg(tmp_path)
    reg = MarbleRegistry()
    reg.create("base", cfg_path=cfg)
    reg.duplicate("base", "copy")
    assert set(reg.list()) == {"base", "copy"}
    assert reg.get("copy") is not reg.get("base")


def test_registry_create_overwrite(tmp_path):
    cfg = create_cfg(tmp_path)
    reg = MarbleRegistry()
    first = reg.create("main", cfg_path=cfg)
    # Creating again without overwrite returns existing instance
    same = reg.create("main", cfg_path=cfg)
    assert first is same
    # Forcing overwrite replaces the instance
    replacement = reg.create("main", cfg_path=cfg, overwrite=True)
    assert replacement is not first

