# tests/test_data.py
"""Tests for corpus loading, sanitization, splitting, and batch sampling."""
import os
import tempfile
import torch
import pytest


SAMPLE_CORPUS = (
    "User: Hello\n"
    "Parhi: Hi there!\n\n"
    "User: How are you?\n"
    "Parhi: I\u2019m doing great!\n"  # curly apostrophe
)


@pytest.fixture
def corpus_file(tmp_path):
    p = tmp_path / "corpus.txt"
    p.write_text(SAMPLE_CORPUS, encoding="utf-8")
    return str(p)


def test_load_corpus_returns_clean_string(corpus_file):
    from data import load_corpus

    text = load_corpus(corpus_file)
    assert isinstance(text, str)
    assert len(text) > 0
    # curly apostrophe should be normalized to straight
    assert "\u2019" not in text
    assert "'" in text


def test_load_corpus_strips_edges(corpus_file):
    from data import load_corpus

    text = load_corpus(corpus_file)
    assert text == text.strip()


def test_build_splits_ratio():
    from data import build_splits

    data = torch.arange(100)
    train, val = build_splits(data, train_frac=0.9)
    assert len(train) == 90
    assert len(val) == 10


def test_get_batch_shapes():
    from config import ParhiConfig
    from data import build_splits, get_batch

    cfg = ParhiConfig(block_size=8, batch_size=4, device="cpu")
    data = torch.randint(0, 50, (200,))
    train, val = build_splits(data, train_frac=0.9)
    x, y = get_batch("train", train_data=train, val_data=val, config=cfg)
    assert x.shape == (4, 8)
    assert y.shape == (4, 8)
    assert x.device.type == "cpu"


def test_get_batch_target_is_shifted_input():
    from config import ParhiConfig
    from data import build_splits, get_batch

    cfg = ParhiConfig(block_size=8, batch_size=2, device="cpu")
    data = torch.arange(100)
    train, val = build_splits(data, train_frac=0.9)
    x, y = get_batch("train", train_data=train, val_data=val, config=cfg)
    # y should be x shifted right by 1 in the original data
    assert y.dtype == x.dtype


def test_get_batch_raises_on_short_data():
    """Fix #4: get_batch must raise ValueError if data is too short."""
    from config import ParhiConfig
    from data import get_batch

    cfg = ParhiConfig(block_size=100, batch_size=2, device="cpu")
    short_data = torch.arange(50)  # shorter than block_size
    with pytest.raises(ValueError, match="must be greater than"):
        get_batch("train", train_data=short_data, val_data=short_data, config=cfg)
