import marble_brain
from marble_brain import Brain


class DummyNB:
    def __init__(self):
        self.calls = []
        self.global_activation_count = 0

    def train(self, data, epochs=1, dream_buffer=None):
        self.calls.append((list(data), epochs))

    def modulate_plasticity(self, ctx):
        pass


class DummyCore:
    def __init__(self, pre_epochs=0, min_k=1):
        self.params = {"pretraining_epochs": pre_epochs, "min_cluster_k": min_k}
        self.neurons = []
        self.synapses = []
        self.cluster_calls = 0

    def get_usage_by_tier(self, tier):
        return 0.0

    def cluster_neurons(self, k):
        self.cluster_called_with = k
        self.cluster_calls += 1

    def relocate_clusters(self, high, medium):
        pass


class _DummyPbar(list):
    def __init__(self, iterable):
        super().__init__(iterable)

    def set_postfix(self, *a, **k):
        pass

    def close(self):
        pass


marble_brain.tqdm = lambda x, **k: _DummyPbar(x)


def test_pretraining_epochs_runs_once():
    core = DummyCore(pre_epochs=2)
    nb = DummyNB()
    brain = Brain(core, nb, None, metrics_visualizer=None, dream_enabled=False)
    brain.train([(0.1, 0.2)], epochs=1)
    assert nb.calls[0] == ([(0.1, 0.1)], 2)
    assert nb.calls[1] == ([(0.1, 0.2)], 1)
    brain.train([(0.2, 0.3)], epochs=1)
    assert nb.calls[2] == ([(0.2, 0.3)], 1)


def test_min_cluster_k_enforced():
    core = DummyCore(min_k=3)
    nb = DummyNB()
    brain = Brain(
        core,
        nb,
        None,
        metrics_visualizer=None,
        dream_enabled=False,
        cluster_k=2,
        auto_cluster_interval=1,
    )
    brain.train([(0.1, 0.2)], epochs=1)
    assert core.cluster_called_with == 3


def test_auto_cluster_interval_respected():
    core = DummyCore()
    nb = DummyNB()
    brain = Brain(
        core,
        nb,
        None,
        metrics_visualizer=None,
        dream_enabled=False,
        auto_cluster_interval=2,
    )
    brain.train([(0.1, 0.2)], epochs=5)
    assert core.cluster_calls == 2
