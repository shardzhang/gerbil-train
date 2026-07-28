"""MF (Matrix Factorization) model for CTR prediction.

1-MF learns user and item embeddings and predicts via dot product.

$$ \hat{y} = \text{sigmoid}(bias + \mathbf{u}_u \cdot \mathbf{v}_i) $$
"""

from __future__ import annotations

from typing import Mapping

import torch
from torch import Tensor, nn

from gerbil_train.config.model_config import MFModelConfig, FieldEntry
from gerbil_train.models.base_model import BaseModel

__all__ = ["MF"]


class MF(BaseModel):
    """Matrix Factorization model for CTR prediction."""

    def __init__(self, model_cfg: MFModelConfig) -> None:
        super().__init__()

        self._validate_fields(model_cfg)

        self.fields_cfg: Mapping[str, FieldEntry] = model_cfg.embedding_fields
        mf_cfg = model_cfg.mf
        user_field = mf_cfg.get("user_field", "user_id")
        item_field = mf_cfg.get("item_field", "item_id")
        embedding_dim = int(mf_cfg.get("embedding_dim", 8))

        if user_field not in self.fields_cfg:
            raise ValueError(f"user_field '{user_field}' not found in embedding_fields")
        if item_field not in self.fields_cfg:
            raise ValueError(f"item_field '{item_field}' not found in embedding_fields")

        self.user_field = user_field
        self.item_field = item_field
        user_entry = self.fields_cfg[user_field]
        item_entry = self.fields_cfg[item_field]

        # User and item embeddings (regular Embedding, not bag)
        self.user_embedding = nn.Embedding(
            num_embeddings=int(user_entry.dim),
            embedding_dim=embedding_dim,
        )
        self.item_embedding = nn.Embedding(
            num_embeddings=int(item_entry.dim),
            embedding_dim=embedding_dim,
        )

        self.global_bias = nn.Parameter(torch.zeros(1))
        self.reset_parameters()

    def _validate_fields(self, model_cfg: MFModelConfig) -> None:
        if not model_cfg.embedding_fields:
            raise ValueError("embedding_fields must be a non-empty mapping")

    def reset_parameters(self) -> None:
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)
        nn.init.zeros_(self.global_bias)

    def _extract_last_index(self, bag: Mapping[str, Tensor]) -> Tensor:
        """Extract the last index of each bag (skipping the sentinel 0)."""
        offsets = bag["offsets"]
        indices = bag["indices"]
        ends = torch.cat([offsets[1:], offsets.new_tensor([indices.size(0)])])
        return indices[ends - 1]

    def forward(self, feature_bags: Mapping[str, Mapping[str, Tensor]]) -> Tensor:
        device = next(self.parameters()).device

        # Extract user and item indices (skip sentinel)
        user_idx = self._extract_last_index(feature_bags[self.user_field]).to(device)
        item_idx = self._extract_last_index(feature_bags[self.item_field]).to(device)

        # Look up embeddings
        user_emb = self.user_embedding(user_idx)
        item_emb = self.item_embedding(item_idx)

        # MF prediction: sigmoid(bias + user·item)
        scores = (user_emb * item_emb).sum(dim=-1) + self.global_bias
        return torch.sigmoid(scores)
