import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from semi_supervised_pairs_pipeline import SemiSupervisedPairsPipeline
from bit_tensor_dataset import BitTensorDataset
from marble_core import Core, DataLoader
from tokenizer_utils import built_in_tokenizer
from tests.test_core_functions import minimal_params


def test_semi_supervised_pipeline_trains(tmp_path):
    params = minimal_params()
    core = Core(params)
    save_path = tmp_path / "semi.pkl"
    pipeline = SemiSupervisedPairsPipeline(core, save_path=str(save_path))
    labeled = [(0.0, 1.0), (0.5, 0.5)]
    unlabeled = [0.1, 0.2]
    out_path = pipeline.train(labeled, unlabeled, epochs=1)
    assert out_path == str(save_path)
    assert save_path.is_file()


def test_semi_supervised_pipeline_non_numeric(tmp_path):
    params = minimal_params()
    core = Core(params)
    save_path = tmp_path / "semi.pkl"
    pipeline = SemiSupervisedPairsPipeline(core, save_path=str(save_path))
    labeled = [("a", "b"), ("c", "d")]
    unlabeled = ["x", "y"]
    pipeline.train(labeled, unlabeled, epochs=1)
    assert save_path.is_file()


def test_semi_supervised_pipeline_with_tokenizer(tmp_path):
    params = minimal_params()
    core = Core(params)
    text_file = tmp_path / "train.txt"
    text_file.write_text("hello world")
    tok = built_in_tokenizer("bert_wordpiece", lowercase=True)
    tok.train([str(text_file)], vocab_size=20)
    dl = DataLoader(tokenizer=tok)
    save_path = tmp_path / "tok_semi.pkl"
    pipeline = SemiSupervisedPairsPipeline(core, save_path=str(save_path), dataloader=dl)
    labeled = [("hello", "world"), ("foo", "bar")]
    unlabeled = ["baz", "qux"]
    pipeline.train(labeled, unlabeled, epochs=1)
    assert save_path.is_file()


def test_semi_supervised_pipeline_bit_dataset(tmp_path):
    params = minimal_params()
    core = Core(params)
    save_path = tmp_path / "bit_semi.pkl"
    pipeline = SemiSupervisedPairsPipeline(core, save_path=str(save_path))
    labeled = [("hi", "there"), ("foo", "bar")]
    unlabeled = ["x", "y"]
    labeled_ds = BitTensorDataset(labeled)
    unlabeled_ds = BitTensorDataset([(u, u) for u in unlabeled])
    pipeline.train(labeled_ds, unlabeled_ds, epochs=1)
    assert save_path.is_file()


def test_semi_supervised_pipeline_auto_bit_dataset(tmp_path):
    params = minimal_params()
    core = Core(params)
    save_path = tmp_path / "auto_semi.pkl"
    pipeline = SemiSupervisedPairsPipeline(core, save_path=str(save_path), use_vocab=True)
    labeled = [("a", "b"), ("c", "d")]
    unlabeled = ["e", "f"]
    pipeline.train(labeled, unlabeled, epochs=1)
    assert save_path.is_file()
    assert isinstance(pipeline.labeled_dataset, BitTensorDataset)
    assert pipeline.labeled_dataset.get_vocab() is not None
