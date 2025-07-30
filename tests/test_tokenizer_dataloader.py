import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pytest
from marble import DataLoader
from tokenizer_utils import (
    built_in_tokenizer,
    tokenizer_to_json,
    tokenizer_from_json,
)


def test_dataloader_tokenizer_roundtrip(tmp_path):
    text_file = tmp_path / "train.txt"
    text_file.write_text("hello world\nhello marble")
    tok = built_in_tokenizer("bert_wordpiece", lowercase=True)
    tok.train([str(text_file)], vocab_size=20)
    dl = DataLoader(tokenizer=tok)
    text = "hello marble"
    encoded = dl.encode(text)
    decoded = dl.decode(encoded)
    assert decoded == text

def test_checkpoint_preserves_tokenizer(tmp_path):
    from marble_core import Core
    from marble_neuronenblitz import Neuronenblitz
    from marble_brain import Brain
    from tests.test_core_functions import minimal_params

    text_file = tmp_path / "train.txt"
    text_file.write_text("hello world")
    tok = built_in_tokenizer("bert_wordpiece", lowercase=True)
    tok.train([str(text_file)], vocab_size=20)
    dl = DataLoader(tokenizer=tok)

    core = Core(minimal_params())
    nb = Neuronenblitz(core)
    brain = Brain(core, nb, dl, save_dir=str(tmp_path))
    ckpt = tmp_path / "ckpt.pkl"
    brain.save_checkpoint(str(ckpt), epoch=1)

    new_dl = DataLoader()
    new_brain = Brain(core, nb, new_dl, save_dir=str(tmp_path))
    new_brain.load_checkpoint(str(ckpt))
    assert new_brain.dataloader.tokenizer is not None
    encoded = new_brain.dataloader.encode("hello world")
    decoded = new_brain.dataloader.decode(encoded)
    assert decoded == "hello world"


@pytest.mark.parametrize(
    "name",
    [
        "bert_wordpiece",
        "byte_level_bpe",
        "char_bpe",
        "sentencepiece_bpe",
        "sentencepiece_unigram",
    ],
)
def test_all_builtin_tokenizers_roundtrip(tmp_path, name):
    text_file = tmp_path / "train.txt"
    text_file.write_text("hello world\nhello marble")
    tok = built_in_tokenizer(name)
    tok.train([str(text_file)], vocab_size=20)
    dl = DataLoader(tokenizer=tok)
    encoded = dl.encode("hello marble")
    decoded = dl.decode(encoded)
    assert decoded == "hello marble"
    json_data = tokenizer_to_json(tok)
    tok2 = tokenizer_from_json(json_data)
    dl2 = DataLoader(tokenizer=tok2)
    assert dl2.decode(dl2.encode("hello")) == "hello"
