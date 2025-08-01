import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from bit_tensor_dataset import BitTensorDataset


def test_roundtrip_no_vocab():
    data = [(123, {"a": 1}), (456, [1, 2, 3])]
    ds = BitTensorDataset(data)
    assert len(ds) == 2
    t_in, t_out = ds[0]
    obj_in = ds.tensor_to_object(t_in)
    obj_out = ds.tensor_to_object(t_out)
    assert obj_in == data[0][0]
    assert obj_out == data[0][1]


def test_roundtrip_with_vocab():
    data = [("hello", "world"), ("foo", "bar")]
    ds = BitTensorDataset(data, use_vocab=True)
    t_in, t_out = ds[1]
    obj_in = ds.tensor_to_object(t_in)
    obj_out = ds.tensor_to_object(t_out)
    assert obj_in == data[1][0]
    assert obj_out == data[1][1]
    assert ds.get_vocab() is not None


def test_vocab_options_max_size_and_occurrence():
    data = [("foo", "bar"), ("foo", "bar")]
    ds = BitTensorDataset(
        data,
        use_vocab=True,
        max_vocab_size=1,
        min_occurrence=1000,
    )
    assert ds.get_vocab() == {}


def test_vocab_only_mode_changes_output():
    data = [("a", "b"), ("c", "d"), ("e", "f")]
    ds = BitTensorDataset(data, use_vocab=True, mixed=False, max_vocab_size=1)
    t_in, _ = ds[0]
    with pytest.raises(Exception):
        ds.tensor_to_object(t_in)


def test_custom_vocab_reuse():
    data = [("x", "y")]
    ds1 = BitTensorDataset(data, use_vocab=True)
    vocab = ds1.get_vocab()
    ds2 = BitTensorDataset(data, vocab=vocab)
    assert ds2.get_vocab() == vocab
    obj = ds2.tensor_to_object(ds2[0][0])
    assert obj == "x"


def test_max_word_length_respected():
    data = [("abcd", "efgh"), ("ijkl", "mnop")]
    ds = BitTensorDataset(data, use_vocab=True, max_word_length=3)
    vocab = ds.get_vocab()
    assert all(len(pattern) <= 3 for pattern in vocab)


def test_dataset_device_setting():
    data = [(1, 2)]
    ds = BitTensorDataset(data, device="cpu")
    assert ds[0][0].device == torch.device("cpu")
    obj = ds.tensor_to_object(ds[0][0])
    assert obj == 1


def test_bit_tensor_dataset_compression():
    data = [("long string" * 10, 123)]
    ds = BitTensorDataset(data, compress=True)
    encoded, _ = ds[0]
    decoded = ds.tensor_to_object(encoded)
    assert decoded == data[0][0]


def test_bit_tensor_dataset_iteration_and_save_load(tmp_path):
    pairs = [(1, 2), (3, 4)]
    ds = BitTensorDataset(pairs, start_id=500)
    assert list(ds) == [ds[0], ds[1]]
    save_path = tmp_path / "ds.pt"
    ds.save(save_path)
    loaded = BitTensorDataset.load(save_path)
    assert len(loaded) == 2
    assert loaded.tensor_to_object(loaded[0][0]) == 1
    assert loaded.start_id == 500


def test_bit_tensor_dataset_summary():
    pairs = [(1, 2), (3, 4)]
    ds = BitTensorDataset(pairs, use_vocab=True)
    info = ds.summary()
    assert info["num_pairs"] == 2
    assert info["vocab_size"] == ds.vocab_size()
    assert info["device"] == str(ds.device)
    assert info["compressed"] is False
    assert info["start_id"] == ds.start_id


def test_bit_tensor_dataset_add_extend():
    ds = BitTensorDataset([(0, 1)])
    ds.add_pair(2, 3)
    assert len(ds) == 2
    ds.extend([(4, 5), (6, 7)])
    assert len(ds) == 4
    assert ds.tensor_to_object(ds[2][0]) == 4


def test_bit_tensor_dataset_custom_start_id():
    data = [("aa", "bb"), ("cc", "dd")]
    ds = BitTensorDataset(data, use_vocab=True, start_id=700)
    vocab_vals = list(ds.get_vocab().values())
    assert vocab_vals and min(vocab_vals) >= 700
    assert ds.start_id == 700


def test_bit_tensor_dataset_iter_decoded_and_summary():
    data = [(1, 2), (3, 4)]
    ds = BitTensorDataset(data)
    decoded = list(ds.iter_decoded())
    assert decoded == data
    info = ds.summary()
    total = sum(a.numel() + b.numel() for a, b in ds)
    expected_avg = float(total) / len(ds)
    assert info["avg_pair_length"] == expected_avg


def test_bit_tensor_dataset_json_roundtrip():
    data = [("x", "y"), ("a", "b")]
    ds = BitTensorDataset(data, use_vocab=True, compress=True, start_id=300)
    json_str = ds.to_json()
    clone = BitTensorDataset.from_json(json_str)
    assert list(clone.iter_decoded()) == data
    assert clone.start_id == 300
    assert clone.get_vocab() == ds.get_vocab()
    info = clone.summary()
    assert "total_bytes" in info and info["total_bytes"] > 0


def test_bit_tensor_dataset_map_and_filter():
    data = [(1, 2), (3, 4)]
    ds = BitTensorDataset(data)
    ds.map_pairs(lambda a, b: (a * 2, b * 2))
    assert ds.tensor_to_object(ds[0][0]) == 2
    ds.filter_pairs(lambda a, b: a > 2)
    assert len(ds) == 1
    assert ds.tensor_to_object(ds[0][0]) == 6


def test_bit_tensor_dataset_split_shuffle_and_hash():
    pairs = [(i, i + 1) for i in range(10)]
    ds = BitTensorDataset(pairs)
    ds.shuffle(generator=torch.Generator().manual_seed(0))
    first, second = ds.split(0.6, shuffle=False)
    assert len(first) == 6
    assert len(second) == 4
    assert ds.hash() == BitTensorDataset.from_json(ds.to_json()).hash()


def test_bit_tensor_dataset_collate_fn():
    data = [("a", "b" * 2), ("long", "c")]
    ds = BitTensorDataset(data, use_vocab=True)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=2, collate_fn=BitTensorDataset.collate_fn
    )
    batch = next(iter(loader))
    inp, out = batch
    assert inp.ndim == 3 and out.ndim == 3
    assert inp.shape[0] == 2 and out.shape[0] == 2


def test_split_deterministic_consistent():
    pairs = [(i, i + 1) for i in range(10)]
    ds1 = BitTensorDataset(pairs)
    train1, val1, test1 = ds1.split_deterministic(0.6, 0.2, salt="seed")

    ds2 = BitTensorDataset(list(reversed(pairs)))
    train2, val2, test2 = ds2.split_deterministic(0.6, 0.2, salt="seed")

    assert list(train1.iter_decoded()) == list(train2.iter_decoded())
    assert list(val1.iter_decoded()) == list(val2.iter_decoded())
    assert list(test1.iter_decoded()) == list(test2.iter_decoded())


def test_dataset_merge(tmp_path):
    a = BitTensorDataset([(1, "a"), (2, "b")])
    b = BitTensorDataset([(2, "c"), (3, "d")])
    merged = a.merge(b, prefer="self")
    assert len(merged) == 3
    assert dict(merged.iter_decoded())[2] == "b"
    merged_other = a.merge(b, prefer="other")
    assert dict(merged_other.iter_decoded())[2] == "c"


def test_dataset_cached(tmp_path):
    path = tmp_path / "cache.pt"
    data1 = [(1, 2), (3, 4)]
    ds1 = BitTensorDataset.cached(data1, path)
    assert path.exists()

    data2 = [(5, 6)]
    ds2 = BitTensorDataset.cached(data2, path)
    assert len(ds2) == len(ds1)
    assert ds2.hash() == ds1.hash()


def test_add_stream_pair(tmp_path):
    import http.server
    import socketserver
    import threading

    content = b"stream-data"

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)

    with socketserver.TCPServer(("localhost", 0), Handler) as httpd:
        port = httpd.server_address[1]
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()

        url = f"http://localhost:{port}"
        ds = BitTensorDataset([])
        ds.add_stream_pair(url, url)

        httpd.shutdown()
        thread.join()

    assert len(ds) == 1
    inp, tgt = ds[0]
    assert ds.tensor_to_object(inp) == content
    assert ds.tensor_to_object(tgt) == content


def test_dataset_deduplicate():
    data = [(1, 2), (1, 2), (3, 4)]
    ds = BitTensorDataset(data)
    ds.deduplicate()
    assert len(ds) == 2
    assert list(ds.iter_decoded()) == [(1, 2), (3, 4)]


def test_dataset_index_lookup():
    data = [(10, 20), (30, 40)]
    ds = BitTensorDataset(data)
    h = ds.hash_pair(1)
    pair = ds.get_by_hash(h)
    assert ds.tensor_to_object(pair[0]) == 30
    assert ds.tensor_to_object(pair[1]) == 40


def test_dataset_checksum_verification(tmp_path):
    ds = BitTensorDataset([(5, 6)])
    path = tmp_path / "ds.pt"
    ds.save(path)
    obj = torch.load(path)
    a, b = obj["data"][0]
    obj["data"][0] = (a + 1, b)
    torch.save(obj, path)
    with pytest.raises(ValueError):
        BitTensorDataset.load(path)


def test_add_stream_pair_async(tmp_path):
    import http.server
    import socketserver
    import threading
    import asyncio

    content = b"async-data"

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)

    with socketserver.TCPServer(("localhost", 0), Handler) as httpd:
        port = httpd.server_address[1]
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()

        url = f"http://localhost:{port}"
        ds = BitTensorDataset([])

        async def run():
            await ds.add_stream_pair_async(url, url)

        asyncio.run(run())

        httpd.shutdown()
        thread.join()

    assert len(ds) == 1
    inp, tgt = ds[0]
    assert ds.tensor_to_object(inp) == content
    assert ds.tensor_to_object(tgt) == content


def test_prune_invalid():
    ds = BitTensorDataset([("ok", "1"), ("bad", "2")])
    ds.data[1] = (
        torch.randint(0, 2, (5, 8), dtype=torch.uint8),
        torch.randint(0, 2, (5, 8), dtype=torch.uint8),
    )
    removed = ds.prune_invalid()
    assert removed == 1
    assert len(ds) == 1
    assert ds.tensor_to_object(ds[0][0]) == "ok"
