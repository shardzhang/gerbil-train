"""SLIM (Sparse Linear Method) for Top-N Recommendation.

Implements the SLIM algorithm from:
    Ning & Karypis, "SLIM: Sparse Linear Methods for Top-N Recommender Systems", ICDM 2011.

Core formula (eq. 1):
    score(user, item_j) = sum_{k in I_user} a_uk · w_kj

W is learned via elastic net (eq. 4):
    min ½||a_j - A·w_j||²₂ + ½·β·||w_j||²₂ + λ·||w_j||₁
    s.t. w_j >= 0, w_jj = 0

Neural approximation:
    w_kj ≈ dot(emb_k, emb_j)  -- item-item similarity via shared embedding
    a_uk = 1 (binary interaction, naturally weighted by history counts)
    prediction = sigmoid(bias + score(user, item_j))
"""

from __future__ import annotations

from typing import Mapping, Any

import torch
from torch import Tensor, nn

from gerbil_train.config.model_config import BaseModelConfig, FieldEntry
from gerbil_train.utils.embedding import bag_to_padded, embed_one_field, to_device
from gerbil_train.models.base_model import BaseModel

__all__ = ["SLIM"]


class SLIM(BaseModel):
    """SLIM model for Top-N recommendation."""

    def __init__(self, model_cfg: BaseModelConfig) -> None:
        super().__init__()

        raw_cfg: dict[str, Any] = getattr(model_cfg, "slim", {})
        self.embedding_fields: Mapping[str, FieldEntry] = model_cfg.embedding_fields
        self.history_field: str = raw_cfg["history_field"]
        self.target_field: str = raw_cfg["target_field"]
        self.embedding_dim: int = int(raw_cfg["embedding_dim"])
        target_entry = self.embedding_fields[self.target_field]

        # Shared item embedding table (target and history items share same vocabulary)
        self.item_embedding = nn.EmbeddingBag(
            num_embeddings=int(target_entry.dim),
            embedding_dim=self.embedding_dim,
            mode="sum",
        )
        self.bias = nn.Parameter(torch.zeros(1))
        self.reset_parameters()
        self._validate_fields(model_cfg)
        

    def _validate_fields(self, model_cfg: BaseModelConfig) -> None:
        if not model_cfg.embedding_fields:
            raise ValueError("embedding_fields must be a non-empty mapping")
        if self.history_field not in self.embedding_fields:
            raise ValueError(f"history_field '{self.history_field}' not in embedding_fields")
        if self.target_field not in self.embedding_fields:
            raise ValueError(f"target_field '{self.target_field}' not in embedding_fields")

    def reset_parameters(self) -> None:
        nn.init.normal_(self.item_embedding.weight, std=0.01)
        nn.init.zeros_(self.bias)

    def _extract_target_index(self, bag: Mapping[str, Tensor]) -> Tensor:
        """Extract the target item index from its bag (skip sentinel 0)."""
        offsets = bag["offsets"]
        indices = bag["indices"]
        ends = torch.cat([offsets[1:], offsets.new_tensor([indices.size(0)])])
        return indices[ends - 1]

    def forward(self, feature_bags: Mapping[str, Mapping[str, Tensor]]) -> Tensor:
        device = next(self.parameters()).device

        # 1. Target item embedding
        target_idx = self._extract_target_index(feature_bags[self.target_field])
        target_emb = to_device(self.item_embedding.weight, device)[target_idx]  # [B, k]

        # 2. History items: convert bag to padded sequences
        hist_bag = feature_bags[self.history_field]
        hist_ids, hist_weights, lengths, max_seq_len = bag_to_padded(hist_bag, device)

        # Look up history item embeddings
        hist_emb = to_device(self.item_embedding.weight, device)[hist_ids]  # [B, L, k]

        # 3. SLIM similarity: dot product of each history item with target item
        # w_kj ≈ dot(emb_k, emb_j) as neural approximation of eq. 1
        sims = (hist_emb * target_emb.unsqueeze(1)).sum(dim=-1)  # [B, L]

        # Mask invalid positions (beyond actual sequence length)
        mask = torch.arange(max_seq_len, device=device).unsqueeze(0) < lengths.unsqueeze(1)
        sims = sims.masked_fill(~mask, 0.0)

        # 4. Aggregate: sum over all history items (eq. 1: sum of a_uk * w_kj)
        score = sims.sum(dim=-1)  # [B]

        logits = score + self.bias
        return torch.sigmoid(logits)
