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

