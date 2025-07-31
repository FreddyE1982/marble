import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from autoencoder_pipeline import AutoencoderPipeline
from bit_tensor_dataset import BitTensorDataset
from marble_core import Core, DataLoader
from tokenizer_utils import built_in_tokenizer
from tests.test_core_functions import minimal_params


def test_autoencoder_pipeline_trains(tmp_path):
    params = minimal_params()
    core = Core(params)
    save_path = tmp_path / "auto.pkl"
    pipeline = AutoencoderPipeline(core, save_path=str(save_path))
    data = [0.1, 0.2, 0.3]
    out_path = pipeline.train(data, epochs=1)
    assert out_path == str(save_path)
    assert save_path.is_file()


def test_autoencoder_pipeline_non_numeric(tmp_path):
    params = minimal_params()
    core = Core(params)
    save_path = tmp_path / "auto.pkl"
    pipeline = AutoencoderPipeline(core, save_path=str(save_path))
    data = ["a", "b"]
    pipeline.train(data, epochs=1)
    assert save_path.is_file()


def test_autoencoder_pipeline_with_tokenizer(tmp_path):
    params = minimal_params()
    core = Core(params)
    text_file = tmp_path / "train.txt"
    text_file.write_text("hello world")
    tok = built_in_tokenizer("bert_wordpiece", lowercase=True)
    tok.train([str(text_file)], vocab_size=20)
    dl = DataLoader(tokenizer=tok)
    save_path = tmp_path / "tok_auto.pkl"
    pipeline = AutoencoderPipeline(core, save_path=str(save_path), dataloader=dl)
    data = ["hello", "world"]
    pipeline.train(data, epochs=1)
    assert save_path.is_file()


def test_autoencoder_pipeline_bit_dataset(tmp_path):
    params = minimal_params()
    core = Core(params)
    save_path = tmp_path / "bit_auto.pkl"
    pipeline = AutoencoderPipeline(core, save_path=str(save_path))
    data = ["hi", "there"]
    ds = BitTensorDataset([(d, d) for d in data])
    pipeline.train(ds, epochs=1)
    assert save_path.is_file()


def test_autoencoder_pipeline_auto_bit_dataset(tmp_path):
    params = minimal_params()
    core = Core(params)
    save_path = tmp_path / "auto_vocab.pkl"
    pipeline = AutoencoderPipeline(core, save_path=str(save_path), use_vocab=True)
    data = ["a", "b", "c"]
    pipeline.train(data, epochs=1)
    assert save_path.is_file()
    assert isinstance(pipeline.last_dataset, BitTensorDataset)
    assert pipeline.last_dataset.get_vocab() is not None
