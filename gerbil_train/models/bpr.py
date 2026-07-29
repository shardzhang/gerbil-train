"""BPR (Bayesian Personalized Ranking) for implicit feedback recommendation.

BPR optimizes a pairwise ranking loss: for a user u, positive item i should be
ranked higher than negative item j.

    Loss = -ln σ(score_ui - score_uj)

The model uses simple Matrix Factorization:
    score_ui = ⟨user_emb_u, item_emb_i⟩

Reference: https://arxiv.org/abs/1205.2618 (UAI 2009)
"""

from __future__ import annotations

from typing import Mapping, Any

import torch
from torch import Tensor, nn

from gerbil_train.config.model_config import BaseModelConfig, FieldEntry
from gerbil_train.utils.embedding import embed_one_field
from gerbil_train.models.base_model import BaseModel

__all__ = ["BPR"]


class BPR(BaseModel):
    """Bayesian Personalized Ranking via Matrix Factorization."""

    def __init__(self, model_cfg: BaseModelConfig) -> None:
        super().__init__()

        self.embedding_fields: Mapping[str, FieldEntry] = model_cfg.embedding_fields

        # Designate user/item fields from mlp config
        bpr_cfg = dict(model_cfg.mlp)
        self.user_field = str(bpr_cfg.pop("user_field", next(iter(self.embedding_fields))))
        self.item_field = str(bpr_cfg.pop("item_field", next(iter(self.embedding_fields))))
        self.emb_size = int(self.embedding_fields[self.user_field].emb_size)

        # User embedding bag (single-valued, mode="sum" = direct lookup)
        user_entry = self.embedding_fields[self.user_field]
        self.user_embedding_bag = nn.EmbeddingBag(
            num_embeddings=int(user_entry.dim),
            embedding_dim=int(user_entry.emb_size),
            mode="sum",
        )

        # Item embedding bag
        item_entry = self.embedding_fields[self.item_field]
        self.item_embedding_bag = nn.EmbeddingBag(
            num_embeddings=int(item_entry.dim),
            embedding_dim=int(item_entry.emb_size),
            mode="sum",
        )
        self._validate_fields(model_cfg)
        self.reset_parameters()

    def _validate_fields(self, model_cfg: BaseModelConfig) -> None:
        if not model_cfg.embedding_fields:
            raise ValueError("embedding_fields must be a non-empty mapping")

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.user_embedding_bag.weight)
        nn.init.xavier_uniform_(self.item_embedding_bag.weight)

    def forward(self, feature_bags: Mapping[str, Mapping[str, Tensor]]) -> Tensor:
        """Compute BPR scores for a batch of (user, item) pairs.

        Expects ``feature_bags[self.user_field]`` and ``feature_bags[self.item_field]``
        to contain ``indices``, ``offsets``, ``weights`` just like other models.

        :return: scores [B]
        """
        device = next(self.parameters()).device

        # [batch_size, emb_size]
        user_emb = embed_one_field(
            self.user_embedding_bag,
            feature_bags[self.user_field]["indices"],
            feature_bags[self.user_field]["offsets"],
            feature_bags[self.user_field]["weights"],
            device=device,
        )
        # [batch_size, emb_size]
        item_emb = embed_one_field(
            self.item_embedding_bag,
            feature_bags[self.item_field]["indices"],
            feature_bags[self.item_field]["offsets"],
            feature_bags[self.item_field]["weights"],
            device=device,
        )
        # [batch_size,]
        return (user_emb * item_emb).sum(dim=-1)

    def predict(self, user_ids: Tensor, item_ids: Tensor) -> Tensor:
        """Direct dot product for arbitrary user-item pairs (single-valued).

        :param user_ids: [N] user indices
        :param item_ids: [N] item indices
        :return: [N] scores
        """
        device = next(self.parameters()).device
        batch_size = user_ids.size(0)
        offsets = torch.arange(batch_size, device=device)
        ones = torch.ones(batch_size, device=device)
        user_emb = self.user_embedding_bag(user_ids, offsets, per_sample_weights=ones)
        item_emb = self.item_embedding_bag(item_ids, offsets, per_sample_weights=ones)
        return (user_emb * item_emb).sum(dim=-1)
