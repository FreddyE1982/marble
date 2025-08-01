from dataset_loader import load_dataset
from marble_base import MetricsVisualizer
from memory_manager import MemoryManager


def test_dataset_events(tmp_path):
    path = tmp_path / "d.csv"
    path.write_text("input,target\n1,2\n")
    vis = MetricsVisualizer(track_memory_usage=False, track_cpu_usage=False)
    mem = MemoryManager()
    load_dataset(str(path), metrics_visualizer=vis, memory_manager=mem)
    assert any(e[0] == "dataset_load_start" for e in vis.events)
    assert any(e[0] == "dataset_load_end" for e in vis.events)
