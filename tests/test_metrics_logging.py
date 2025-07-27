from marble_base import MetricsVisualizer


def test_metrics_visualizer_logging(tmp_path):
    tb_dir = tmp_path / "tb"
    csv_file = tmp_path / "metrics.csv"
    mv = MetricsVisualizer(log_dir=str(tb_dir), csv_log_path=str(csv_file))
    mv.update({"loss": 1.0, "vram_usage": 2.0})
    mv.close()
    assert any(p.name.startswith("events.out.tfevents") for p in tb_dir.iterdir())
    assert csv_file.exists() and csv_file.read_text().count("\n") >= 2
