from dataset_loader import load_kuzu_graph, load_training_data_from_config
from kuzu_interface import KuzuGraphDatabase


def build_graph(path):
    db = KuzuGraphDatabase(path)
    db.create_node_table(
        "Sample",
        {"id": "INT64", "input": "DOUBLE", "target": "DOUBLE"},
        "id",
    )
    db.add_node("Sample", {"id": 1, "input": 0.1, "target": 0.2})
    db.add_node("Sample", {"id": 2, "input": 0.2, "target": 0.4})
    return db


def test_load_kuzu_graph(tmp_path):
    db_path = tmp_path / "train.kuzu"
    db = build_graph(str(db_path))
    db.close()
    pairs = load_kuzu_graph(
        str(db_path),
        "MATCH (s:Sample) RETURN s.input AS input, s.target AS target",
    )
    assert pairs == [(0.1, 0.2), (0.2, 0.4)]


def test_load_training_data_from_config_kuzu(tmp_path):
    db_path = tmp_path / "train.kuzu"
    db = build_graph(str(db_path))
    db.close()
    cfg = {
        "use_kuzu_graph": True,
        "kuzu_graph": {
            "db_path": str(db_path),
            "query": "MATCH (s:Sample) RETURN s.input AS input, s.target AS target",
        },
    }
    pairs = load_training_data_from_config(cfg)
    assert pairs == [(0.1, 0.2), (0.2, 0.4)]
