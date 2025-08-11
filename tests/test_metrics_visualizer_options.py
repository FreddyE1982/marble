import matplotlib.pyplot as plt
from marble_base import MetricsVisualizer


def test_metrics_visualizer_style_and_annotations():
    mv = MetricsVisualizer(color_scheme="dark_background", dpi=123, show_neuron_ids=True)
    assert mv.fig.get_dpi() == 123
    mv.update({"loss": 1.0, "vram_usage": 0.5})
    # Show neuron ids should annotate the plot with text labels
    assert mv.ax.texts, "Expected annotations when show_neuron_ids is True"
    # Dark style applies a black axes facecolor
    assert plt.rcParams["axes.facecolor"] == "black"
    mv.close()
