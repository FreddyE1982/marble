import sys, os; sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from preprocessing_pipeline import PreprocessingPipeline
from tokenizer_utils import built_in_tokenizer
from marble import DataLoader


def test_preprocessing_pipeline(tmp_path):
    data = [1, 2, 3]
    steps = [lambda x: x * 2, lambda x: x + 1]
    cache_dir = tmp_path / "cache"
    pp = PreprocessingPipeline(steps, cache_dir=str(cache_dir))
    result1 = pp.process(data, dataset_id="test")
    assert result1 == [3, 5, 7]
    # Run again to use cache
    result2 = pp.process(data, dataset_id="test")
    assert result2 == result1


def test_preprocessing_pipeline_with_tokenizer(tmp_path):
    steps = [lambda x: x]
    cache_dir = tmp_path / "tok_cache"
    text_file = tmp_path / "train.txt"
    text_file.write_text("hello world")
    tok = built_in_tokenizer("bert_wordpiece", lowercase=True)
    tok.train([str(text_file)], vocab_size=20)
    dl = DataLoader(tokenizer=tok)
    pp = PreprocessingPipeline(steps, cache_dir=str(cache_dir), dataloader=dl)
    data = ["hello", "world"]
    result1 = pp.process(data, dataset_id="tok")
    assert result1 == ["hello", "world"]
    result2 = pp.process(data, dataset_id="tok")
    assert result2 == result1
