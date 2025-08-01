from dataset_loader import load_dataset
from memory_manager import MemoryManager


def test_memory_manager_notification(tmp_path):
    path = tmp_path / "d.csv"
    path.write_text("input,target\n1,2\n")
    mgr = MemoryManager()
    data = load_dataset(str(path), memory_manager=mgr)
    assert len(data) == 1
    assert mgr.total_reserved() > 0
