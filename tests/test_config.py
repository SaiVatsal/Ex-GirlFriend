# tests/test_config.py
"""Tests for PrarthanaConfig dataclass and device detection."""
import pytest


def test_config_defaults():
    from config import PrarthanaConfig

    cfg = PrarthanaConfig()
    assert cfg.block_size == 128
    assert cfg.batch_size == 32
    assert cfg.n_embd == 192
    assert cfg.n_head == 4
    assert cfg.n_layer == 4
    assert cfg.dropout == pytest.approx(0.2)
    assert cfg.learning_rate == pytest.approx(5e-4)
    assert cfg.weight_decay == pytest.approx(0.01)
    assert cfg.max_iters == 3_000
    assert cfg.eval_interval == 200
    assert cfg.eval_iters == 50


def test_config_head_dim_divisibility():
    from config import PrarthanaConfig

    cfg = PrarthanaConfig()
    assert cfg.n_embd % cfg.n_head == 0, "d_model must be divisible by n_head"


def test_detect_device_returns_string():
    from config import detect_device

    device = detect_device()
    assert device in ("cuda", "mps", "cpu")


def test_config_override():
    from config import PrarthanaConfig

    cfg = PrarthanaConfig(block_size=64, n_embd=96, n_head=3)
    assert cfg.block_size == 64
    assert cfg.n_embd == 96
    assert cfg.n_head == 3
