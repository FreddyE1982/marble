from marble_core import Core
from marble_graph_builder import add_fully_connected_layer, add_neuron_group
from topology_kuzu import TopologyKuzuTracker
from tests.test_core_functions import minimal_params


def test_live_topology_updates(tmp_path):
    db_path = tmp_path / "topo_db"
    core = Core(minimal_params(), formula="0", formula_num_neurons=0)
    tracker = TopologyKuzuTracker(core, str(db_path))

    add_neuron_group(core, 2)
    rows = tracker.db.execute("MATCH (n:Neuron) RETURN count(n) AS cnt")
    assert rows[0]["cnt"] == 2

    add_fully_connected_layer(core, [0], 1, weights=[[0.5]])
    rows = tracker.db.execute("MATCH ()-[r:SYNAPSE]->() RETURN count(r) AS cnt")
    assert rows[0]["cnt"] == 1

    core.increase_representation_size(1)
    rows = tracker.db.execute("MATCH (n:Neuron {id:0}) RETURN n.rep_size AS rs")
    assert rows[0]["rs"] == core.rep_size
