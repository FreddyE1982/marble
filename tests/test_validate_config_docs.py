from scripts.validate_config_docs import validate_config_docs


def test_validate_config_docs_detects_discrepancies(tmp_path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text("a:\n  b: 1\n", encoding="utf-8")
    doc = tmp_path / "CONFIGURABLE_PARAMETERS.md"
    doc.write_text("## a\n- c\n", encoding="utf-8")
    missing, extra = validate_config_docs(cfg, doc)
    assert missing == ["a.b"]
    assert extra == ["a.c"]

def test_validate_config_docs_no_discrepancy(tmp_path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text("a:\n  b: 1\n", encoding="utf-8")
    doc = tmp_path / "CONFIGURABLE_PARAMETERS.md"
    doc.write_text("## a\n- b\n", encoding="utf-8")
    missing, extra = validate_config_docs(cfg, doc)
    assert missing == []
    assert extra == []
