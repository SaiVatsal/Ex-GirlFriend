# data.py
"""Corpus ingestion, sanitization, train/val splitting, and batch sampling."""
from __future__ import annotations

import re
import unicodedata

import torch

from config import ParhiConfig


def load_corpus(path: str) -> str:
    """Read a plaintext corpus file and apply sanitization.

    Sanitization steps:
      1. Strip leading/trailing whitespace.
      2. Normalize Unicode to NFC form.
      3. Replace common curly quotes and apostrophes with ASCII equivalents.
      4. Collapse runs of 3+ newlines into 2.

    Args:
        path: Filesystem path to the corpus text file.

    Returns:
        The cleaned corpus as a single string.
    """
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    text = text.strip()
    text = unicodedata.normalize("NFC", text)

    # Curly quotes -> straight
    replacements: dict[str, str] = {
        "\u2018": "'",   # left single curly
        "\u2019": "'",   # right single curly
        "\u201C": '"',   # left double curly
        "\u201D": '"',   # right double curly
        "\u2013": "-",   # en dash
        "\u2014": "--",  # em dash
        "\u2026": "...", # ellipsis
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)

    # Collapse excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def build_splits(
    data: torch.Tensor,
    train_frac: float = 0.9,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split a 1-D token tensor into training and validation subsets.

    Args:
        data: 1-D tensor of token IDs.
        train_frac: Fraction of data allocated to training.

    Returns:
        (train_data, val_data) -- two non-overlapping 1-D tensors.
    """
    n = int(train_frac * len(data))
    return data[:n], data[n:]


def get_batch(
    split: str,
    *,
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    config: ParhiConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample a random batch of input-target pairs for language modelling.

    For each sample in the batch, a random starting index is chosen and
    a window of ``config.block_size`` tokens is extracted as input (X).
    The target (Y) is the same window shifted right by one position.

    Args:
        split: One of ``'train'`` or ``'val'``.
        train_data: 1-D tensor of training token IDs.
        val_data: 1-D tensor of validation token IDs.
        config: Hyperparameter configuration.

    Returns:
        ``(X, Y)`` each of shape ``(batch_size, block_size)`` on ``config.device``.

    Raises:
        ValueError: If the split data is shorter than ``block_size + 1``.
    """
    data = train_data if split == "train" else val_data
    max_start = len(data) - config.block_size - 1
    if max_start <= 0:
        raise ValueError(
            f"{split}_data length ({len(data)}) must be greater than "
            f"block_size ({config.block_size}) + 1. "
            "Please expand the corpus or reduce block_size."
        )
    ix = torch.randint(0, max_start, (config.batch_size,))
    x = torch.stack([data[i : i + config.block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + 1 + config.block_size] for i in ix])
    return x.to(config.device), y.to(config.device)
