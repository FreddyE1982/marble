import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from torrent_offload import BrainTorrentTracker, BrainTorrentClient
import pytest


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
    # simulate unstable connection by stopping its worker
    target_client._stop_worker()

    # simulate unstable connection by stopping its worker
    target_client._stop_worker()

    # simulate unstable connection by stopping its worker
    target_client._stop_worker()

    # simulate unstable connection by stopping its worker
    target_client._stop_worker()
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
    brain = Brain(core, nb, DataLoader(), torrent_client=client_main, torrent_map=torrent_map, torrent_offload_enabled=True)
    brain.lobe_manager.genesis(range(len(core.neurons)))
    brain.offload_high_attention_torrent(threshold=-1.0)

    out, _ = nb.dynamic_wander(0.2)
    assert isinstance(out, float)


def test_torrent_async_and_buffering():
    from marble_core import Core
    from tests.test_core_functions import minimal_params
    import time

    tracker = BrainTorrentTracker()
    client_main = BrainTorrentClient('main', tracker, buffer_size=2)
    client_peer = BrainTorrentClient('peer', tracker, buffer_size=2)

    client_main.connect()
    client_peer.connect()

    core = Core(minimal_params())
    part = client_main.offload(core)
    assigned = tracker.part_to_client[part]
    target_client = tracker.clients[assigned]
    # simulate unstable connection by stopping its worker
    target_client._stop_worker()

    # slow down processing to trigger queue usage
    original = target_client.process

    def slow_process(value, p):
        time.sleep(0.2)
        return original(value, p)

    target_client.process = slow_process

    fut1 = target_client.process_async(0.1, part)
    fut2 = target_client.process_async(0.2, part)
    with pytest.raises(BufferError):
        target_client.process_async(0.3, part, timeout=0.1)

    # restart worker to process buffered tasks
    target_client._start_worker()

    out1 = fut1.result(timeout=2)
    out2 = fut2.result(timeout=2)
    assert isinstance(out1, float)
    assert isinstance(out2, float)

    target_client.process = original

