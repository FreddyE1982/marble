import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from imitation_pairs_pipeline import ImitationPairsPipeline
from bit_tensor_dataset import BitTensorDataset
from marble_core import Core, DataLoader
from tokenizer_utils import built_in_tokenizer
from tests.test_core_functions import minimal_params


def test_imitation_pairs_pipeline_trains(tmp_path):
    params = minimal_params()
    core = Core(params)
    save_path = tmp_path / "imit.pkl"
    pipeline = ImitationPairsPipeline(core, save_path=str(save_path))
    pairs = [(0.1, 0.2), (0.3, 0.4)]
    out_path = pipeline.train(pairs, epochs=1)
    assert out_path == str(save_path)
    assert save_path.is_file()


def test_imitation_pairs_pipeline_non_numeric(tmp_path):
    params = minimal_params()
    core = Core(params)
    save_path = tmp_path / "imit.pkl"
    pipeline = ImitationPairsPipeline(core, save_path=str(save_path))
    pairs = [("hello", "greet"), ("world", "noun")]
    pipeline.train(pairs, epochs=1)
    assert save_path.is_file()


def test_imitation_pairs_pipeline_with_tokenizer(tmp_path):
    params = minimal_params()
    core = Core(params)
    text_file = tmp_path / "train.txt"
    text_file.write_text("hello world")
    tok = built_in_tokenizer("bert_wordpiece", lowercase=True)
    tok.train([str(text_file)], vocab_size=20)
    dl = DataLoader(tokenizer=tok)
    save_path = tmp_path / "tok_imit.pkl"
    pipeline = ImitationPairsPipeline(core, save_path=str(save_path), dataloader=dl)
    pairs = [("hello", "world"), ("foo", "bar")]
    pipeline.train(pairs, epochs=1)
    assert save_path.is_file()


def test_imitation_pairs_pipeline_bit_dataset(tmp_path):
    params = minimal_params()
    core = Core(params)
    save_path = tmp_path / "bit_imit.pkl"
    pipeline = ImitationPairsPipeline(core, save_path=str(save_path))
    pairs = [("hi", "there"), ("foo", "bar")]
    ds = BitTensorDataset(pairs)
    pipeline.train(ds, epochs=1)
    assert save_path.is_file()


def test_imitation_pairs_pipeline_auto_bit_dataset(tmp_path):
    params = minimal_params()
    core = Core(params)
    save_path = tmp_path / "auto_imit.pkl"
    pipeline = ImitationPairsPipeline(core, save_path=str(save_path), use_vocab=True)
    pairs = [("a", "b"), ("c", "d")]
    pipeline.train(pairs, epochs=1)
    assert save_path.is_file()
    assert isinstance(pipeline.last_dataset, BitTensorDataset)
    assert pipeline.last_dataset.get_vocab() is not None
