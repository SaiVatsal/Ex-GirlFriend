# train.py
"""Parhi-GPT training engine with validation and checkpoint persistence.

Usage:
    python train.py

Reads ``parhi_corpus.txt``, builds a character-level vocabulary,
constructs the Transformer, and trains with AdamW.  Checkpoints are
saved to ``parhi_model.pt`` and resumed automatically if present.
"""
from __future__ import annotations

import dataclasses
import os
import sys
import time

import torch

from config import ParhiConfig, detect_device
from tokenizer import CharTokenizer
from data import load_corpus, build_splits, get_batch
from model import ParhiGPT


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
CORPUS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "parhi_corpus.txt")
CHECKPOINT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "parhi_model.pt")
VOCAB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vocab.json")


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
@torch.no_grad()
def estimate_loss(
    model: ParhiGPT,
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    config: ParhiConfig,
) -> dict[str, float]:
    """Estimate mean cross-entropy loss over several batches.

    Args:
        model: The language model in eval mode.
        train_data: 1-D tensor of training token IDs.
        val_data: 1-D tensor of validation token IDs.
        config: Hyperparameter configuration.

    Returns:
        Dict with keys ``'train'`` and ``'val'``, each mapping to
        the mean loss (float).
    """
    model.eval()
    losses: dict[str, float] = {}
    for split in ("train", "val"):
        total = 0.0
        for _ in range(config.eval_iters):
            x, y = get_batch(
                split,
                train_data=train_data,
                val_data=val_data,
                config=config,
            )
            _, loss = model(x, y)
            total += loss.item()
        losses[split] = total / config.eval_iters
    model.train()
    return losses


def save_checkpoint(
    path: str,
    model: ParhiGPT,
    optimizer: torch.optim.Optimizer,
    tokenizer: CharTokenizer,
    step: int,
    best_val_loss: float,
    config: ParhiConfig | None = None,
) -> None:
    """Persist model, optimizer, vocab, config, and metadata to disk.

    Args:
        path: Destination filepath for the checkpoint.
        model: The language model.
        optimizer: The optimizer.
        tokenizer: Character tokenizer with populated stoi/itos.
        step: Current training step.
        best_val_loss: Lowest recorded validation loss so far.
        config: Optional ParhiConfig to persist (fix #5).
    """
    payload: dict = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "vocab_stoi": tokenizer.stoi,
        "vocab_itos": tokenizer.itos,
        "step": step,
        "best_val_loss": best_val_loss,
    }
    if config is not None:
        payload["config"] = dataclasses.asdict(config)
    torch.save(payload, path)


def load_checkpoint(
    path: str,
    model: ParhiGPT,
    optimizer: torch.optim.Optimizer,
    device: str,
) -> dict:
    """Restore model and optimizer state from a checkpoint.

    Args:
        path: Source filepath for the checkpoint.
        model: An already-instantiated model (weights will be overwritten).
        optimizer: An already-instantiated optimizer (state will be overwritten).
        device: Target device string.

    Returns:
        Dict with ``'step'``, ``'best_val_loss'``, ``'vocab_stoi'``,
        ``'vocab_itos'``, and optionally ``'config'`` entries.
    """
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    result = {
        "step": ckpt["step"],
        "best_val_loss": ckpt["best_val_loss"],
        "vocab_stoi": ckpt["vocab_stoi"],
        "vocab_itos": ckpt["vocab_itos"],
    }
    if "config" in ckpt:
        result["config"] = ckpt["config"]
    return result


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------
def train() -> None:
    """Full training entry point for Parhi-GPT.

    1. Load & sanitize corpus.
    2. Build character-level tokenizer.
    3. Split data 90/10.
    4. Construct model + AdamW optimizer.
    5. Optionally resume from checkpoint.
    6. Train for ``max_iters`` steps, evaluating and checkpointing
       every ``eval_interval`` steps.
    """
    print("=" * 60)
    print("  Parhi-GPT -- Training Engine")
    print("=" * 60)

    # --- Config ---
    config = ParhiConfig()
    print(f"\n[device] {config.device}")

    # --- Corpus ---
    if not os.path.exists(CORPUS_PATH):
        print(f"[error] corpus not found at {CORPUS_PATH}")
        sys.exit(1)

    text = load_corpus(CORPUS_PATH)
    print(f"[corpus] {len(text):,} characters loaded")

    # --- Tokenizer ---
    tokenizer = CharTokenizer()
    tokenizer.build(text)
    tokenizer.save(VOCAB_PATH)
    config.vocab_size = tokenizer.vocab_size
    print(f"[vocab] {config.vocab_size} unique characters")

    # --- Encode & split ---
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    train_data, val_data = build_splits(data)
    print(f"[split] train={len(train_data):,}  val={len(val_data):,}")

    # --- Model ---
    model = ParhiGPT(config).to(config.device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] {n_params:,} parameters")

    # --- Optimizer ---
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    # --- Resume ---
    start_step = 0
    best_val_loss = float("inf")
    if os.path.exists(CHECKPOINT_PATH):
        print(f"[resume] loading checkpoint from {CHECKPOINT_PATH}")
        meta = load_checkpoint(CHECKPOINT_PATH, model, optimizer, config.device)
        start_step = meta["step"]
        best_val_loss = meta["best_val_loss"]
        print(f"[resume] step={start_step}, best_val_loss={best_val_loss:.4f}")

    # --- Training ---
    print(f"\n[train] starting from step {start_step} -> {config.max_iters}")
    model.train()
    t0 = time.time()

    for step in range(start_step, config.max_iters):
        # Evaluation checkpoint
        if step % config.eval_interval == 0 or step == config.max_iters - 1:
            losses = estimate_loss(model, train_data, val_data, config)
            elapsed = time.time() - t0
            print(
                f"  step {step:>6d} | "
                f"train {losses['train']:.4f} | "
                f"val {losses['val']:.4f} | "
                f"elapsed {elapsed:.1f}s"
            )
            if losses["val"] < best_val_loss:
                best_val_loss = losses["val"]
                save_checkpoint(
                    CHECKPOINT_PATH, model, optimizer, tokenizer,
                    step, best_val_loss, config,
                )
                print(f"  -> checkpoint saved (val_loss={best_val_loss:.4f})")

        # Forward + backward
        x, y = get_batch(
            "train",
            train_data=train_data,
            val_data=val_data,
            config=config,
        )
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

    # --- Final save ---
    save_checkpoint(
        CHECKPOINT_PATH, model, optimizer, tokenizer,
        config.max_iters, best_val_loss, config,
    )
    total_time = time.time() - t0
    print(f"\n[done] training complete in {total_time:.1f}s")
    print(f"[done] best val loss: {best_val_loss:.4f}")
    print(f"[done] checkpoint: {CHECKPOINT_PATH}")


if __name__ == "__main__":
    train()
