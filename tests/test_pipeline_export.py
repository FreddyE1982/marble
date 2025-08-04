from marble_core import Core
from marble_neuronenblitz import Neuronenblitz
from pipeline import Pipeline
from tests.test_core_functions import minimal_params


def test_automatic_export(tmp_path):
    params = minimal_params()
    core = Core(params)
    nb = Neuronenblitz(core)
    pipe = Pipeline()
    export_file = tmp_path / "model.json"
    results = pipe.execute(marble=nb, export_path=str(export_file))
    assert export_file.exists() and export_file.stat().st_size > 0
    assert str(export_file) in results
