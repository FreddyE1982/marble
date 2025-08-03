"""Utilities for working with Hugging Face tokenizers."""

from __future__ import annotations

from typing import Iterable, Iterator

from tokenizers import Tokenizer
from tokenizers.models import BPE, WordPiece, Unigram
from tokenizers.normalizers import NFD, Lowercase, StripAccents, Sequence
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer, WordPieceTrainer, UnigramTrainer
from tokenizers import (
    BertWordPieceTokenizer,
    ByteLevelBPETokenizer,
    CharBPETokenizer,
    SentencePieceBPETokenizer,
    SentencePieceUnigramTokenizer,
)


def load_tokenizer(path: str) -> Tokenizer:
    """Load a tokenizer from a JSON file."""
    return Tokenizer.from_file(path)


def built_in_tokenizer(name: str, **kwargs) -> Tokenizer:
    """Instantiate a built-in tokenizer by name."""
    mapping = {
        "bert_wordpiece": BertWordPieceTokenizer,
        "byte_level_bpe": ByteLevelBPETokenizer,
        "char_bpe": CharBPETokenizer,
        "sentencepiece_bpe": SentencePieceBPETokenizer,
        "sentencepiece_unigram": SentencePieceUnigramTokenizer,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported tokenizer: {name}")
    return mapping[name](**kwargs)


def train_tokenizer(files: Iterable[str], model: str, vocab_size: int = 30000, **kwargs) -> Tokenizer:
    """Train a tokenizer model from text files."""
    if model == "wordpiece":
        tokenizer = Tokenizer(WordPiece())
        tokenizer.normalizer = Sequence([NFD(), Lowercase(), StripAccents()])
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordPieceTrainer(vocab_size=vocab_size, **kwargs)
    elif model == "bpe":
        tokenizer = Tokenizer(BPE())
        tokenizer.pre_tokenizer = Whitespace()
        trainer = BpeTrainer(vocab_size=vocab_size, **kwargs)
    elif model == "unigram":
        tokenizer = Tokenizer(Unigram())
        trainer = UnigramTrainer(vocab_size=vocab_size, **kwargs)
    else:
        raise ValueError(f"Unsupported model: {model}")
    tokenizer.train(files, trainer)
    return tokenizer


def tokenizer_to_json(tokenizer: Tokenizer) -> str:
    """Return the JSON representation of ``tokenizer``."""
    return tokenizer.to_str()


def tokenizer_from_json(data: str) -> Tokenizer:
    """Load a tokenizer from a JSON string."""
    return Tokenizer.from_str(data)


def tokenize_line(tokenizer: Tokenizer, line: str) -> list[int]:
    """Tokenize a single ``line`` and return token ids.

    This utility avoids loading an entire corpus into memory and enables
    streaming tokenisation.
    """

    return tokenizer.encode(line).ids


def tokenize_lines(tokenizer: Tokenizer, lines: Iterable[str]) -> Iterator[list[int]]:
    """Yield token ids for each line in ``lines`` using ``tokenizer``."""

    for line in lines:
        yield tokenize_line(tokenizer, line)
