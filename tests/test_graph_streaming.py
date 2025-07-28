import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tests.test_core_functions import minimal_params
from marble_core import Core
from graph_streaming import stream_graph_chunks, identify_memory_hotspots, benchmark_streaming


def test_stream_graph_chunks_covers_all():
    params = minimal_params()
    core = Core(params)
    core.expand(num_new_neurons=10, num_new_synapses=0)
    for i in range(9):
        core.add_synapse(i, i + 1)

    chunks = list(stream_graph_chunks(core, 3))
    assert sum(len(n) for n, _ in chunks) == len(core.neurons)
    assert sum(len(s) for _, s in chunks) >= len(core.synapses)


def test_identify_memory_hotspots_nonzero():
    params = minimal_params()
    core = Core(params)
    core.expand(num_new_neurons=1, num_new_synapses=0)
    usage = identify_memory_hotspots(core)
    assert usage["neurons"] > 0
    assert usage["synapses"] >= 0


def test_benchmark_streaming_runs():
    params = minimal_params()
    core = Core(params)
    rate = benchmark_streaming(core, 1, iters=1)
    assert rate >= 0
