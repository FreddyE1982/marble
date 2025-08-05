from graph_viz import sankey_figure


def test_sankey_filters():
    data = {
        "nodes": [{"id": 0}, {"id": 1}, {"id": 2}],
        "edges": [
            {"source": 0, "target": 1, "weight": 0.5},
            {"source": 1, "target": 2, "weight": 0.1},
        ],
    }
    fig = sankey_figure(data, weight_threshold=0.2, degree_threshold=0)
    assert len(fig.data[0]["link"]["value"]) == 1
    fig2 = sankey_figure(data, weight_threshold=0.0, degree_threshold=2)
    assert len(fig2.data[0]["link"]["value"]) == 0
