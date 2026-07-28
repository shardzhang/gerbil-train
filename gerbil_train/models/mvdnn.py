"""MV-DNN (Multi-View Deep Neural Network) for retrieval/CTR.

MV-DNN learns separate user and item representations by projecting them
into a shared embedding space via independent DNNs, then computing cosine
similarity.

    score(u, i) = cosine(user_dnn(f_user), item_dnn(f_item))

Reference: https://www.microsoft.com/en-us/research/publication/multi-view-deep-neural-network-for-cross-view-learning/
"""

from __future__ import annotations

from typing import Mapping, Any

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from gerbil_train.config.model_config import BaseModelConfig, FieldEntry
from gerbil_train.utils.embedding import embed_one_field
from gerbil_train.models.layers import FullyConnectedLayer
from gerbil_train.models.base_model import BaseModel

__all__ = ["MVDNN"]


class MVDNN(BaseModel):
    """Multi-View Deep Neural Network for retrieval."""

    def __init__(self, model_cfg: BaseModelConfig) -> None:
        super().__init__()

        self.embedding_fields: Mapping[str, FieldEntry] = model_cfg.embedding_fields
        self.field_names = list(self.embedding_fields.keys())

        # User/item fields from config
        mv_cfg: dict[str, Any] = model_cfg.mlp
        self.user_field = str(mv_cfg["user_field"])
        self.item_field = str(mv_cfg["item_field"])
        self.user_side_fields = list(mv_cfg.get("user_side_fields", []))
        self.item_side_fields = list(mv_cfg.get("item_side_fields", []))
        self.emb_size = int(mv_cfg.get("embedding_dim", 16))

        # Collect all user fields + user side info fields
        self.all_user_fields = [self.user_field] + self.user_side_fields
        self.all_item_fields = [self.item_field] + self.item_side_fields

        # Embedding bags (shared by field_index)
        self.embedding_bags = nn.ModuleDict()
        for field_name, entry in self.embedding_fields.items():
            if entry.field_type == 0 and entry.concat_type == "direct":
                continue
            key = str(entry.field_index)
            if key not in self.embedding_bags:
                self.embedding_bags[key] = nn.EmbeddingBag(
                    num_embeddings=int(entry.dim),
                    embedding_dim=int(entry.emb_size),
                    mode="sum",
                )

        # Compute user/item input dims
        def _view_dim(field_names: list[str]) -> int:
            d = 0
            for fn in field_names:
                entry = self.embedding_fields[fn]
                if entry.field_type == 0 and entry.concat_type == "direct":
                    d += int(entry.dim)
                else:
                    d += int(entry.emb_size)
            return d

        user_dim = _view_dim(self.all_user_fields)
        item_dim = _view_dim(self.all_item_fields)

        # User tower
        user_hidden = list(mv_cfg.get("user_hidden", [128, 64]))
        self.user_tower = FullyConnectedLayer(
            input_dim=user_dim, hidden_dims=user_hidden,
            bias=[True] * len(user_hidden),
            batch_norm=True, activation="relu", dropout=0.1,
        )

        # Item tower
        item_hidden = list(mv_cfg.get("item_hidden", [128, 64]))
        self.item_tower = FullyConnectedLayer(
            input_dim=item_dim, hidden_dims=item_hidden,
            bias=[True] * len(item_hidden),
            batch_norm=True, activation="relu", dropout=0.1,
        )

        # Projection to shared embedding space
        self.user_proj = nn.Linear(user_hidden[-1], self.emb_size)
        self.item_proj = nn.Linear(item_hidden[-1], self.emb_size)

        self._validate_fields(model_cfg)
        self.reset_parameters()

    def _validate_fields(self, model_cfg: BaseModelConfig) -> None:
        if not model_cfg.embedding_fields:
            raise ValueError("embedding_fields must be a non-empty mapping")
        if "user_field" not in model_cfg.mlp or "item_field" not in model_cfg.mlp:
            raise ValueError("MV-DNN config must specify user_field and item_field")

    def reset_parameters(self) -> None:
        for emb in self.embedding_bags.values():
            nn.init.xavier_uniform_(emb.weight)

    def _embed_view(self, field_names: list[str], feature_bags: dict) -> Tensor:
        """Embed and concat all fields for a view."""
        embs: list[Tensor] = []
        device = next(self.parameters()).device
        for fn in field_names:
            entry = self.embedding_fields[fn]
            if entry.field_type == 0 and entry.concat_type == "direct":
                embs.append(feature_bags[fn]["weights"].view(-1, int(entry.dim)).to(device))
            else:
                embs.append(embed_one_field(
                    self.embedding_bags[str(entry.field_index)],
                    feature_bags[fn]["indices"],
                    feature_bags[fn]["offsets"],
                    feature_bags[fn]["weights"],
                    device=device,
                ))
        return torch.cat(embs, dim=-1)

    def forward(self, feature_bags: Mapping[str, Mapping[str, Tensor]]) -> Tensor:
        """Compute score for (user, item) pairs.

        :return: [B] cosine similarity scores in [-1, 1]
        """
        user_input = self._embed_view(self.all_user_fields, feature_bags)
        item_input = self._embed_view(self.all_item_fields, feature_bags)

        user_emb = F.normalize(self.user_proj(self.user_tower(user_input)), dim=-1)
        item_emb = F.normalize(self.item_proj(self.item_tower(item_input)), dim=-1)

        return (user_emb * item_emb).sum(dim=-1)

    def encode_user(self, feature_bags: Mapping[str, Mapping[str, Tensor]]) -> Tensor:
        """Extract user embedding (for ANN retrieval)."""
        user_input = self._embed_view(self.all_user_fields, feature_bags)
        return F.normalize(self.user_proj(self.user_tower(user_input)), dim=-1)

    def encode_item(self, feature_bags: Mapping[str, Mapping[str, Tensor]]) -> Tensor:
        """Extract item embedding (for ANN retrieval)."""
        item_input = self._embed_view(self.all_item_fields, feature_bags)
        return F.normalize(self.item_proj(self.item_tower(item_input)), dim=-1)
