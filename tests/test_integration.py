# tests/test_integration.py
"""End-to-end integration test: corpus -> tokenizer -> model -> train step -> generate."""
import os
import torch
import pytest


def test_full_pipeline(tmp_path):
    """Smoke test: load corpus, tokenize, train 5 steps, generate text."""
    from config import ParhiConfig
    from tokenizer import CharTokenizer
    from data import load_corpus, build_splits, get_batch
    from model import ParhiGPT
    from train import estimate_loss, save_checkpoint, load_checkpoint

    # 1. Write a mini corpus
    corpus_path = str(tmp_path / "corpus.txt")
    with open(corpus_path, "w", encoding="utf-8") as f:
        f.write("User: hello\nParhi: hi there!\n" * 20)

    # 2. Load & tokenize
    text = load_corpus(corpus_path)
    tok = CharTokenizer()
    tok.build(text)
    assert tok.vocab_size > 0

    # 3. Encode & split
    data = torch.tensor(tok.encode(text), dtype=torch.long)
    train_data, val_data = build_splits(data)

    # 4. Build model
    cfg = ParhiConfig(
        block_size=16, batch_size=4, n_embd=32, n_head=4,
        n_layer=2, dropout=0.0, max_iters=5, eval_interval=5,
        eval_iters=2, vocab_size=tok.vocab_size, device="cpu",
    )
    model = ParhiGPT(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)

    # 5. Train a few steps
    model.train()
    for _ in range(5):
        x, y = get_batch("train", train_data=train_data, val_data=val_data, config=cfg)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    # 6. Eval
    losses = estimate_loss(model, train_data, val_data, cfg)
    assert losses["train"] > 0
    assert losses["val"] > 0

    # 7. Generate
    model.eval()
    prompt_ids = tok.encode("User:")
    idx = torch.tensor([prompt_ids], dtype=torch.long)
    out = model.generate(idx, max_new_tokens=20, temperature=0.8, top_k=10)
    generated = tok.decode(out[0].tolist())
    assert len(generated) > len("User:")

    # 8. Checkpoint round-trip (fix #5: includes config)
    ckpt_path = str(tmp_path / "ckpt.pt")
    save_checkpoint(ckpt_path, model, optimizer, tok, step=5, best_val_loss=losses["val"], config=cfg)
    assert os.path.exists(ckpt_path)

    model2 = ParhiGPT(cfg)
    opt2 = torch.optim.AdamW(model2.parameters(), lr=cfg.learning_rate)
    meta = load_checkpoint(ckpt_path, model2, opt2, "cpu")
    assert meta["step"] == 5
    assert "config" in meta

    # 9. ParhiCLI
    from chat import ParhiCLI

    cli = ParhiCLI(checkpoint_path=ckpt_path, device="cpu")
    assert cli.config.n_embd == 32  # fix #5 — config from checkpoint
    response = cli.respond("hello")
    assert isinstance(response, str)
