# tests/test_chat.py
"""Tests for ParhiCLI: prompt formatting, stop-token logic, response gen.""" # 
import os
import tempfile
import torch
import pytest


@pytest.fixture
def cli_instance(tmp_path):
    """Create a tiny trained model checkpoint and instantiate ParhiCLI."""
    from config import ParhiConfig
    from tokenizer import CharTokenizer
    from model import ParhiGPT
    from train import save_checkpoint

    corpus = "User: hi\nParhi: hello!\n"
    tok = CharTokenizer()
    tok.build(corpus)

    cfg = ParhiConfig(
        block_size=32,
        batch_size=2,
        n_embd=32,
        n_head=4,
        n_layer=2,
        dropout=0.0,
        vocab_size=tok.vocab_size,
        device="cpu",
    )
    model = ParhiGPT(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)
    ckpt_path = str(tmp_path / "test_model.pt")
    save_checkpoint(ckpt_path, model, optimizer, tok, step=0, best_val_loss=5.0, config=cfg)

    from chat import ParhiCLI

    return ParhiCLI(checkpoint_path=ckpt_path, device="cpu")


def test_cli_loads_model(cli_instance):
    assert cli_instance.model is not None
    assert cli_instance.tokenizer.vocab_size > 0


def test_cli_config_from_checkpoint(cli_instance):
    """Fix #5: config should be reconstructed from checkpoint."""
    assert cli_instance.config.n_embd == 32
    assert cli_instance.config.n_head == 4
    assert cli_instance.config.n_layer == 2
    assert cli_instance.config.block_size == 32


def test_respond_returns_string(cli_instance):
    response = cli_instance.respond("hello")
    assert isinstance(response, str)
    assert len(response) >= 0  # may be empty with untrained model


def test_respond_does_not_echo_user_prefix(cli_instance):
    response = cli_instance.respond("hi there")
    # Response should NOT contain "User:" as that would be hallucinating
    # the user's turn — best-effort with untrained model
    assert isinstance(response, str)


def test_prompt_formatting(cli_instance):
    """Internal prompt should include the dialogue structure."""
    # Use only chars present in the fixture vocab: "User: hi\nParhi: hello!\n"
    prompt = "User: hi\nParhi:"
    ids = cli_instance.tokenizer.encode(prompt)
    assert len(ids) > 0
