import os
from unittest import mock

from huggingface_utils import hf_login, hf_load_dataset, hf_load_model


def test_hf_login_reads_and_writes(tmp_path):
    token_file = tmp_path / "token"
    token_file.write_text("abc")
    with mock.patch("huggingface_utils.login") as login_mock:
        tok = hf_login(token_file=str(token_file))
        login_mock.assert_called_once_with(token="abc")
        assert tok == "abc"

    with mock.patch("huggingface_utils.login") as login_mock:
        tok = hf_login(token="xyz", token_file=str(token_file))
        login_mock.assert_called_once_with(token="xyz")
        assert token_file.read_text() == "xyz"
        assert tok == "xyz"


def test_hf_load_dataset_streaming():
    dummy = [{"input": 1, "target": 2}, {"input": 3, "target": 4}]
    with mock.patch(
        "huggingface_utils.hf_login", return_value=None
    ) as login_mock, mock.patch(
        "huggingface_utils.load_dataset", return_value=dummy
    ) as ld:
        pairs = hf_load_dataset("dummy", "train", streaming=True)
    login_mock.assert_called_once()
    ld.assert_called_once_with("dummy", split="train", token=None, streaming=True)
    assert pairs == [(1, 2), (3, 4)]


def test_hf_load_model():
    with mock.patch("huggingface_utils.hf_login") as login_mock, mock.patch(
        "huggingface_utils.AutoModel.from_pretrained", return_value="m"
    ) as fm:
        model = hf_load_model("bert")
    login_mock.assert_called_once()
    fm.assert_called_once_with("bert", trust_remote_code=True)
    assert model == "m"
