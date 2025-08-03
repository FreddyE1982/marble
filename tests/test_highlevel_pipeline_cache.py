from highlevel_pipeline import HighLevelPipeline
from marble_base import MetricsVisualizer
import torch

counter = {"calls": 0}

def step():
    counter["calls"] += 1
    return counter["calls"]


def test_pipeline_cache(tmp_path):
    mv = MetricsVisualizer()
    hp = HighLevelPipeline(cache_dir=str(tmp_path))
    hp.add_step(step)
    _, res1 = hp.execute(metrics_visualizer=mv)
    assert res1[0] == 1
    assert counter["calls"] == 1
    assert mv.metrics["cache_miss"] == [1]
    _, res2 = hp.execute(metrics_visualizer=mv)
    assert res2[0] == 1
    assert counter["calls"] == 1
    assert mv.metrics["cache_hit"] == [1]
    hp.clear_cache()
    _, _ = hp.execute(metrics_visualizer=mv)
    assert counter["calls"] == 2


def test_default_cache_dir_device():
    hp = HighLevelPipeline()
    expected = "pipeline_cache_gpu" if torch.cuda.is_available() else "pipeline_cache_cpu"
    assert hp.cache_dir == expected
