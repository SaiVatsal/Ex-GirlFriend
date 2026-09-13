# tests/test_train.py
"""Tests for training utilities: loss estimation, checkpoint save/load."""
import dataclasses
import os
import tempfile
import torch
import pytest


@pytest.fixture
def small_setup():
    from config import ParhiConfig
    from tokenizer import CharTokenizer
    from model import ParhiGPT

    tok = CharTokenizer()
    tok.build("abcdefghijklmnopqrstuvwxyz \n:.!?,'-")
    cfg = ParhiConfig(
        block_size=16,
        batch_size=4,
        n_embd=32,
        n_head=4,
        n_layer=2,
        dropout=0.0,
        max_iters=10,
        eval_interval=5,
        eval_iters=2,
        vocab_size=tok.vocab_size,
        device="cpu",
    )
    model = ParhiGPT(cfg)
    return cfg, tok, model


def test_estimate_loss_returns_dict(small_setup):
    from data import build_splits
    from train import estimate_loss

    cfg, tok, model = small_setup
    data = torch.randint(0, cfg.vocab_size, (500,))
    train_data, val_data = build_splits(data)
    result = estimate_loss(model, train_data, val_data, cfg)
    assert "train" in result
    assert "val" in result
    assert isinstance(result["train"], float)
    assert result["train"] > 0


def test_save_and_load_checkpoint(small_setup, tmp_path):
    from train import save_checkpoint, load_checkpoint

    cfg, tok, model = small_setup
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)
    ckpt_path = str(tmp_path / "test_ckpt.pt")

    save_checkpoint(ckpt_path, model, optimizer, tok, step=42, best_val_loss=2.5, config=cfg)
    assert os.path.exists(ckpt_path)

    # Load into a fresh model
    from model import ParhiGPT

    model2 = ParhiGPT(cfg)
    opt2 = torch.optim.AdamW(model2.parameters(), lr=cfg.learning_rate)
    meta = load_checkpoint(ckpt_path, model2, opt2, "cpu")

    assert meta["step"] == 42
    assert meta["best_val_loss"] == pytest.approx(2.5)

    # Weights should match
    for p1, p2 in zip(model.parameters(), model2.parameters()):
        assert torch.allclose(p1, p2)


def test_checkpoint_contains_config(small_setup, tmp_path):
    """Fix #5: checkpoint must store the full config dictionary."""
    from train import save_checkpoint

    cfg, tok, model = small_setup
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)
    ckpt_path = str(tmp_path / "test_ckpt.pt")

    save_checkpoint(ckpt_path, model, optimizer, tok, step=1, best_val_loss=3.0, config=cfg)

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    assert "config" in ckpt
    assert ckpt["config"]["n_embd"] == 32
    assert ckpt["config"]["n_head"] == 4
    assert ckpt["config"]["n_layer"] == 2
    assert ckpt["config"]["block_size"] == 16
