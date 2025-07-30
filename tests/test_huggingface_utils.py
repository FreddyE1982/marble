import huggingface_utils as hfu


def test_auto_hf_login(monkeypatch, tmp_path):
    token = tmp_path / "token"
    token.write_text("tok")
    monkeypatch.setattr(hfu, "HF_TOKEN_PATH", token)
    called = {}

    def fake_login(token=None):
        called["token"] = token

    monkeypatch.setattr(hfu, "hf_login", fake_login)
    hfu._logged_in = False
    hfu.auto_hf_login()
    assert called["token"] == "tok"
    called["token"] = None
    hfu.auto_hf_login()
    assert called["token"] is None


def test_download_hf_model(monkeypatch):
    called = {}

    def fake_download(repo_id, filename, cache_dir=None):
        called["args"] = (repo_id, filename, cache_dir)
        return "/tmp/x"

    monkeypatch.setattr(hfu, "hf_hub_download", fake_download)
    monkeypatch.setattr(hfu, "auto_hf_login", lambda: called.setdefault("login", True))
    path = hfu.download_hf_model("repo", "file")
    assert called["login"]
    assert called["args"] == ("repo", "file", None)
    assert path == "/tmp/x"
