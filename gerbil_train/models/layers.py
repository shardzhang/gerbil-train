"""Reusable neural network layers for model construction."""

from __future__ import annotations

import torch
from torch import Tensor, nn

__all__ = ["ActivationUnit"]


class ActivationUnit(nn.Module):
    """Local Activation Unit (LAU) from the DIN paper.

    Computes attention scores between a target item and each item in a behavior sequence:

        score_i = MLP(concat(behavior_i, target, behavior_i * target))

    :param input_emb_dim: Embedding dimension of both behavior and target items
    :param hidden_dim: Hidden layer size of the attention MLP (default: 36)
    """

    def __init__(self, input_emb_dim: int, hidden_dim: int = 36) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_emb_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, behavior_embs: Tensor, target_emb: Tensor) -> Tensor:
        """Compute raw attention scores for each behavior item.

        :param behavior_embs: ``[total_items, emb_dim]``
        :param target_emb: ``[total_items, emb_dim]``
        :return: ``[total_items]``
        """
        attn_input = torch.cat([behavior_embs, target_emb, behavior_embs * target_emb], dim=-1)
        return self.mlp(attn_input).squeeze(-1)

    def pool(
        self,
        behavior_embs: Tensor,
        target_emb: Tensor,
        lengths: Tensor,
    ) -> Tensor:
        """Attention pooling on padded behavior sequences.

        Accepts already-padded sequences, making it agnostic to the upstream
        data format (TFRecord offsets, packed sequences, etc).

        :param behavior_embs: Padded behavior item embeddings ``[batch, seq_len, emb_dim]``
        :param target_emb: Target item embedding ``[batch, emb_dim]``
        :param lengths: Actual sequence length per sample ``[batch]``
        :return: Interest embedding ``[batch, emb_dim]``
        """
        batch, seq_len, emb_dim = behavior_embs.shape
        target_expanded = target_emb.unsqueeze(1).expand(-1, seq_len, -1)

        scores = self.forward(
            behavior_embs.reshape(-1, emb_dim),
            target_expanded.reshape(-1, emb_dim),
        ).reshape(batch, seq_len)

        mask = torch.arange(seq_len, device=behavior_embs.device).unsqueeze(0) < lengths.unsqueeze(1)
        scores = scores.masked_fill(~mask, -float("inf"))
        has_items = lengths > 0
        scores = scores.masked_fill(~has_items.unsqueeze(1), 0.0)
        weights = torch.softmax(scores, dim=-1)

        return (weights.unsqueeze(-1) * behavior_embs).sum(dim=1)
