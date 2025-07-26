import sys, os; sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from preprocessing_pipeline import PreprocessingPipeline


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
