# tests/test_model.py
"""Tests for the Parhi-GPT Transformer architecture."""
import torch
import pytest


@pytest.fixture
def small_config():
    from config import ParhiConfig

    return ParhiConfig(
        block_size=16,
        batch_size=2,
        n_embd=32,
        n_head=4,
        n_layer=2,
        dropout=0.0,
        vocab_size=50,
        device="cpu",
    )


def test_multihead_attention_output_shape(small_config):
    from model import MultiHeadAttention

    mha = MultiHeadAttention(small_config)
    x = torch.randn(2, 16, 32)
    out = mha(x)
    assert out.shape == (2, 16, 32)


def test_multihead_attention_causal_mask(small_config):
    """Fix #1: Verify mask is -inf above diagonal and 0.0 on/below diagonal."""
    from model import MultiHeadAttention

    mha = MultiHeadAttention(small_config)
    assert hasattr(mha, "mask")
    mask = mha.mask
    T = small_config.block_size
    assert mask.shape == (1, 1, T, T)

    # Strictly upper-triangular positions must be -inf
    triu_idx = torch.triu_indices(T, T, offset=1)
    assert torch.isneginf(mask[0, 0, triu_idx[0], triu_idx[1]]).all()

    # On and below diagonal must be 0.0 (unmasked)
    tril_idx = torch.tril_indices(T, T, offset=0)
    assert (mask[0, 0, tril_idx[0], tril_idx[1]] == 0.0).all()


def test_feedforward_output_shape(small_config):
    from model import FeedForward

    ff = FeedForward(small_config)
    x = torch.randn(2, 16, 32)
    out = ff(x)
    assert out.shape == (2, 16, 32)


def test_transformer_block_output_shape(small_config):
    from model import TransformerBlock

    block = TransformerBlock(small_config)
    x = torch.randn(2, 16, 32)
    out = block(x)
    assert out.shape == (2, 16, 32)


def test_parhi_gpt_forward_logits_shape(small_config):
    from model import ParhiGPT

    model = ParhiGPT(small_config)
    idx = torch.randint(0, 50, (2, 16))
    logits, loss = model(idx)
    assert logits.shape == (2, 16, 50)
    assert loss is None


def test_parhi_gpt_forward_with_targets(small_config):
    from model import ParhiGPT

    model = ParhiGPT(small_config)
    idx = torch.randint(0, 50, (2, 16))
    targets = torch.randint(0, 50, (2, 16))
    logits, loss = model(idx, targets)
    assert logits.shape == (2, 16, 50)
    assert loss is not None
    assert loss.dim() == 0  # scalar
    assert loss.item() > 0


def test_parhi_gpt_generate(small_config):
    from model import ParhiGPT

    model = ParhiGPT(small_config)
    model.eval()
    idx = torch.randint(0, 50, (1, 5))
    out = model.generate(idx, max_new_tokens=10, temperature=0.8, top_k=10)
    assert out.shape == (1, 15)  # 5 prompt + 10 generated
    assert (out[:, :5] == idx).all()  # prompt preserved


def test_parhi_gpt_generate_with_stop_ids(small_config):
    """Fix #3: generate() should stop early when a stop_id is produced."""
    from model import ParhiGPT

    model = ParhiGPT(small_config)
    model.eval()
    idx = torch.randint(0, 50, (1, 5))
    # Use all possible IDs as stop_ids so it stops on first token
    out = model.generate(
        idx, max_new_tokens=100, temperature=0.8, top_k=10,
        stop_ids=list(range(50)),
    )
    # Should have generated exactly 1 token before stopping
    assert out.shape == (1, 6)


def test_parhi_gpt_parameter_count(small_config):
    from model import ParhiGPT

    model = ParhiGPT(small_config)
    n_params = sum(p.numel() for p in model.parameters())
    assert n_params > 0
    # Rough sanity: 2-layer, 32-dim, 50-vocab should be small
    assert n_params < 500_000


def test_parhi_gpt_long_input_truncated(small_config):
    """Input longer than block_size should be truncated to last block_size tokens."""
    from model import ParhiGPT

    model = ParhiGPT(small_config)
    idx = torch.randint(0, 50, (1, 32))  # 32 > block_size=16
    logits, loss = model(idx)
    # Model should truncate to block_size
    assert logits.shape[1] == 16
