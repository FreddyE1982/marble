import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from torrent_offload import BrainTorrentTracker, BrainTorrentClient


def test_torrent_tracker_distribution_and_redistribution():
    tracker = BrainTorrentTracker()
    client_a = BrainTorrentClient('A', tracker)
    client_b = BrainTorrentClient('B', tracker)

    client_a.connect()
    client_b.connect()

    for i in range(4):
        tracker.add_part(i)

    assert len(tracker.clients) == 2
    parts_a = set(tracker.get_client_parts('A'))
    parts_b = set(tracker.get_client_parts('B'))
    assert parts_a.union(parts_b) == {0, 1, 2, 3}

    client_b.disconnect()

    # all parts should now belong to A
    for i in range(4):
        assert tracker.part_to_client[i] == 'A'
        assert i in client_a.parts


def test_torrent_client_offload_process():
    from marble_core import Core
    from tests.test_core_functions import minimal_params

    tracker = BrainTorrentTracker()
    client_main = BrainTorrentClient('main', tracker)
    client_peer = BrainTorrentClient('peer', tracker)

    client_main.connect()
    client_peer.connect()

    core = Core(minimal_params())
    part = client_main.offload(core)
    assigned = tracker.part_to_client[part]
    target_client = tracker.clients[assigned]
    out = target_client.process(0.5, part)
    assert isinstance(out, float)


def test_torrent_offload_dynamic_wander():
    from marble_core import Core, DataLoader
    from marble_neuronenblitz import Neuronenblitz
    from marble_brain import Brain
    from tests.test_core_functions import minimal_params

    tracker = BrainTorrentTracker()
    client_main = BrainTorrentClient('main', tracker)
    client_peer = BrainTorrentClient('peer', tracker)
    client_main.connect()
    client_peer.connect()

    torrent_map = {}
    core = Core(minimal_params())
    nb = Neuronenblitz(core, torrent_client=client_main, torrent_map=torrent_map)
    brain = Brain(core, nb, DataLoader(), torrent_client=client_main, torrent_map=torrent_map)
    brain.lobe_manager.genesis(range(len(core.neurons)))
    brain.offload_high_attention_torrent(threshold=-1.0)

    out, _ = nb.dynamic_wander(0.2)
    assert isinstance(out, float)

