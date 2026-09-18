# model.py
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import ParhiConfig


class MultiHeadAttention(nn.Module):
    def __init__(self, config: ParhiConfig) -> None:
        super().__init__()
        assert config.n_embd % config.n_head == 0, (
            f"n_embd ({config.n_embd}) must be divisible by n_head ({config.n_head})"
        )
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head

        # efficiency
        self.qkv_proj = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        self.out_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        #  triangle filled with -inf
        mask = torch.full(
            (1, 1, config.block_size, config.block_size),
            float("-inf"),
        )
        mask = torch.triu(mask, diagonal=1)
        self.register_buffer("mask", mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply causal multi-head self-attention.

        Args:
            x: Input tensor of shape ``(B, T, C)``.

        Returns:
            Output tensor of shape ``(B, T, C)``.
        """
        B, T, C = x.shape

        # Compute Q, K, V in one matmul
        qkv = self.qkv_proj(x)  # (B, T, 3C)
        q, k, v = qkv.split(C, dim=2)  # each (B, T, C)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        scale = math.sqrt(self.head_dim)
        attn = (q @ k.transpose(-2, -1)) / scale  # (B, h, T, T)
        attn = attn + self.mask[:, :, :T, :T]
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)

        # Weighted sum of values
        out = attn @ v  # (B, h, T, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        return self.resid_dropout(self.out_proj(out))


class FeedForward(nn.Module):
    """Position-wise feed-forward network with GELU activation.

    FFN(x) = Linear_2( GELU( Linear_1(x) ) ) * Dropout
    Expansion factor: 4x (d_ff = 4 * d_model).
    """

    def __init__(self, config: ParhiConfig) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, config: ParhiConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = MultiHeadAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.ffn = FeedForward(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class ParhiGPT(nn.Module):
    """Causal decoder-only Transformer language model.

    Architecture:
        Token Embedding + Positional Embedding
        -> N x TransformerBlock (Pre-LayerNorm)
        -> Final LayerNorm
        -> Linear LM Head -> logits in R^{B x T x V}
    """

    def __init__(self, config: ParhiConfig) -> None:
        super().__init__()
        self.config = config

        self.token_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)

        self.blocks = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.n_layer)]
        )
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight tying: token embedding weights = LM head weights
        self.token_emb.weight = self.lm_head.weight

        # Initialize weights
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        """Custom weight initialization matching GPT-2 conventions."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Forward pass computing logits and optional cross-entropy loss.

        Args:
            idx: Input token IDs of shape ``(B, T)``.
            targets: Target token IDs of shape ``(B, T)``, or ``None``.

        Returns:
            ``(logits, loss)`` where logits has shape ``(B, T, V)``
            and loss is a scalar tensor or ``None``.
        """
        B, T = idx.shape

        # Truncate to block_size if input exceeds context window
        if T > self.config.block_size:
            idx = idx[:, -self.config.block_size :]
            if targets is not None:
                targets = targets[:, -self.config.block_size :]
            T = self.config.block_size

        # Embeddings
        tok_emb = self.token_emb(idx)  # (B, T, C)
        pos_emb = self.pos_emb(torch.arange(T, device=idx.device))  # (T, C)
        x = self.drop(tok_emb + pos_emb)

        # Transformer blocks
        for block in self.blocks:
            x = block(x)

        # Final LN + LM head
        x = self.ln_f(x)
        logits = self.lm_head(x)  # (B, T, V)

        # Loss computation
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
            )

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 0.8,
        top_k: int = 30,
        stop_ids: list[int] | None = None,
    ) -> torch.Tensor:
        """Autoregressively generate tokens with optional early stopping.

        Args:
            idx: Conditioning token IDs of shape ``(B, T)``.
            max_new_tokens: Number of new tokens to generate.
            temperature: Softmax temperature for sampling.
            top_k: Number of top logits to keep before sampling.
            stop_ids: Optional list of token IDs that trigger early exit
                when generated (e.g., newline character for dialogue).

        Returns:
            Extended token sequence of shape ``(B, T + generated)``.
        """
        for _ in range(max_new_tokens):
            # Crop to block_size
            idx_cond = idx[:, -self.config.block_size :]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-5)  # (B, V)

            # Top-K filtering
            if top_k > 0:
                topk_vals, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                threshold = topk_vals[:, -1].unsqueeze(-1)
                logits[logits < threshold] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)  # (B, 1)
            idx = torch.cat([idx, next_id], dim=1)

            # Immediate early exit on stop token
            if stop_ids and next_id.item() in stop_ids:
                break

        return idx
