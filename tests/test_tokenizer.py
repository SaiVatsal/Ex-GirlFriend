# tests/test_tokenizer.py
"""Tests for CharTokenizer: build, encode, decode, persistence."""
import json
import os
import tempfile
import pytest


def test_build_creates_bijective_mapping():
    from tokenizer import CharTokenizer

    tok = CharTokenizer()
    tok.build("hello world")
    chars = set("hello world")
    assert len(tok.stoi) == len(chars)
    assert len(tok.itos) == len(chars)
    # bijection: every char maps to unique id and back
    for ch in chars:
        assert tok.itos[tok.stoi[ch]] == ch


def test_encode_decode_roundtrip():
    from tokenizer import CharTokenizer

    tok = CharTokenizer()
    tok.build("abcxyz 123")
    text = "xyz abc"
    ids = tok.encode(text)
    assert isinstance(ids, list)
    assert all(isinstance(i, int) for i in ids)
    assert tok.decode(ids) == text


def test_vocab_size_property():
    from tokenizer import CharTokenizer

    tok = CharTokenizer()
    tok.build("abcd")
    assert tok.vocab_size == 4  # 'a', 'b', 'c', 'd'


def test_save_and_load(tmp_path):
    from tokenizer import CharTokenizer

    path = str(tmp_path / "vocab.json")
    tok1 = CharTokenizer()
    tok1.build("hello")
    tok1.save(path)

    tok2 = CharTokenizer()
    tok2.load(path)
    assert tok2.stoi == tok1.stoi
    assert tok2.itos == tok1.itos
    assert tok2.encode("hello") == tok1.encode("hello")


def test_encode_unknown_char_raises():
    from tokenizer import CharTokenizer

    tok = CharTokenizer()
    tok.build("abc")
    with pytest.raises(KeyError):
        tok.encode("z")
