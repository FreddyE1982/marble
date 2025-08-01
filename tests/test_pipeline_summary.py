import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pipeline import Pipeline
from dataset_loader import load_dataset, export_dataset


def test_pipeline_dataset_summary(tmp_path):
    csv_path = tmp_path / "data.csv"
    csv_path.write_text("input,target\n1,2\n3,4\n")
    pipe = Pipeline([
        {"func": "load_dataset", "module": "dataset_loader", "params": {"source": str(csv_path)}},
    ])
    results = pipe.execute()
    assert len(results) == 1
    summaries = pipe.dataset_summaries()
    assert summaries and summaries[0]["num_pairs"] == 2
