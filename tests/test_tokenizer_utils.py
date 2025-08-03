from tokenizer_utils import built_in_tokenizer, tokenize_line, tokenize_lines


def test_tokenize_line_and_lines():
    tok = built_in_tokenizer("char_bpe")
    tok.train_from_iterator(["hello", "world"], vocab_size=10)
    single = tokenize_line(tok, "hello")
    assert isinstance(single, list) and single
    many = list(tokenize_lines(tok, ["hi", "there"]))
    assert len(many) == 2 and all(isinstance(m, list) for m in many)
