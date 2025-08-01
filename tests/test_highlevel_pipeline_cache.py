from highlevel_pipeline import HighLevelPipeline

counter = {"calls": 0}

def step():
    counter["calls"] += 1
    return counter["calls"]


def test_pipeline_cache(tmp_path):
    hp = HighLevelPipeline(cache_dir=str(tmp_path))
    hp.add_step(step)
    _, res1 = hp.execute()
    assert res1[0] == 1
    assert counter["calls"] == 1
    _, res2 = hp.execute()
    assert res2[0] == 1
    assert counter["calls"] == 1
    hp.clear_cache()
    _, _ = hp.execute()
    assert counter["calls"] == 2
