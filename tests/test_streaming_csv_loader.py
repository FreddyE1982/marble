import csv
from dataset_loader import StreamingCSVLoader
from tokenizer_utils import built_in_tokenizer


def test_streaming_csv_loader_resume(tmp_path):
    csv_path = tmp_path / "data.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["input", "target"])
        writer.writerow(["hello", "1"])
        writer.writerow(["world", "2"])

    tok = built_in_tokenizer("char_bpe")
    tok.train_from_iterator(["hello", "world"], vocab_size=10)
    loader = StreamingCSVLoader(str(csv_path), tokenizer=tok)
    it = iter(loader)
    first = next(it)
    assert "input_ids" in first
    loader.close()

    loader2 = StreamingCSVLoader(str(csv_path))
    rows = list(loader2)
    loader2.close()
    assert len(rows) == 1 and rows[0]["input"] == "world"
