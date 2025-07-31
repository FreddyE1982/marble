import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from unified_pairs_pipeline import UnifiedPairsPipeline
from bit_tensor_dataset import BitTensorDataset
from marble_core import Core, DataLoader
from marble_neuronenblitz import Neuronenblitz
from hebbian_learning import HebbianLearner
from autoencoder_learning import AutoencoderLearner
from tokenizer_utils import built_in_tokenizer
from tests.test_core_functions import minimal_params


def _learners(core, nb):
    return {
        "hebb": HebbianLearner(core, nb),
        "auto": AutoencoderLearner(core, nb),
    }


def test_unified_pairs_pipeline_trains(tmp_path):
    params = minimal_params()
    core = Core(params)
    nb = Neuronenblitz(core)
    save_path = tmp_path / "uni.pkl"
    pipeline = UnifiedPairsPipeline(core, _learners(core, nb), save_path=str(save_path))
    pairs = [(0.0, 0.1), (0.2, 0.3)]
    out_path = pipeline.train(pairs, epochs=1)
    assert out_path == str(save_path)
    assert save_path.is_file()


def test_unified_pairs_pipeline_non_numeric(tmp_path):
    params = minimal_params()
    core = Core(params)
    nb = Neuronenblitz(core)
    save_path = tmp_path / "uni.pkl"
    pipeline = UnifiedPairsPipeline(core, _learners(core, nb), save_path=str(save_path))
    pairs = [("hello", "foo"), ("world", "bar")]
    pipeline.train(pairs, epochs=1)
    assert save_path.is_file()


def test_unified_pairs_pipeline_with_tokenizer(tmp_path):
    params = minimal_params()
    core = Core(params)
    nb = Neuronenblitz(core)
    text_file = tmp_path / "train.txt"
    text_file.write_text("hello world")
    tok = built_in_tokenizer("bert_wordpiece", lowercase=True)
    tok.train([str(text_file)], vocab_size=20)
    dl = DataLoader(tokenizer=tok)
    save_path = tmp_path / "tok_uni.pkl"
    pipeline = UnifiedPairsPipeline(core, _learners(core, nb), save_path=str(save_path), dataloader=dl)
    pairs = [("hello", "world"), ("foo", "bar")]
    pipeline.train(pairs, epochs=1)
    assert save_path.is_file()


def test_unified_pairs_pipeline_bit_dataset(tmp_path):
    params = minimal_params()
    core = Core(params)
    nb = Neuronenblitz(core)
    save_path = tmp_path / "bit_uni.pkl"
    pipeline = UnifiedPairsPipeline(core, _learners(core, nb), save_path=str(save_path))
    pairs = [("hi", "there"), ("foo", "bar")]
    ds = BitTensorDataset(pairs)
    pipeline.train(ds, epochs=1)
    assert save_path.is_file()


def test_unified_pairs_pipeline_auto_bit_dataset(tmp_path):
    params = minimal_params()
    core = Core(params)
    nb = Neuronenblitz(core)
    save_path = tmp_path / "auto_uni.pkl"
    pipeline = UnifiedPairsPipeline(core, _learners(core, nb), save_path=str(save_path), use_vocab=True)
    pairs = [("a", "b"), ("c", "d")]
    pipeline.train(pairs, epochs=1)
    assert save_path.is_file()
    assert isinstance(pipeline.last_dataset, BitTensorDataset)
    assert pipeline.last_dataset.get_vocab() is not None
