from highlevel_pipeline import HighLevelPipeline

def step():
    return "x"


def test_pipeline_checkpoint(tmp_path):
    hp = HighLevelPipeline(dataset_version="v1")
    hp.add_step(step)
    path = tmp_path / "chk.pkl"
    hp.save_checkpoint(path)
    loaded = HighLevelPipeline.load_checkpoint(path)
    assert loaded.dataset_version == "v1"
    assert loaded.steps == hp.steps
