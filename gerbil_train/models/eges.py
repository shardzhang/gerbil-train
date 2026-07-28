"""EGES (Enhanced Graph Embedding with Side Information) for CTR prediction.

EGES enhances item embeddings by incorporating side information (category,
brand, etc.) via learned attention weights:

    e_item = attention_weighted_sum(v_base, v_side_1, ..., v_side_K)

Reference: https://doi.org/10.1145/3178876.3186070 (WWW 2018)
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

__all__ = ["EGES"]


class EGES(BaseModel):
    """Enhanced Graph Embedding with Side Information for CTR."""

    def __init__(self, model_cfg: BaseModelConfig) -> None:
        super().__init__()

        self.embedding_fields: Mapping[str, FieldEntry] = model_cfg.embedding_fields
        self.field_names = list(self.embedding_fields.keys())

        # EGES config: item field + side info fields
        eges_cfg: dict[str, Any] = model_cfg.mlp
        self.item_field = str(eges_cfg["item_field"])
        self.side_fields = list(eges_cfg.get("side_fields", []))
        self.emb_size = int(self.embedding_fields[self.item_field].emb_size)

        # All other fields are treated as user/plain features
        self.plain_field_names = [n for n in self.field_names
                                  if n not in ([self.item_field] + self.side_fields)]

        # Embedding bag for each field (shared by field_index)
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

        # Number of attention sources: 1 (base) + K (side info)
        self.num_sources = 1 + len(self.side_fields)

        # Side info projection (if emb_size differs)
        self.side_proj = nn.ModuleDict()
        for sf in self.side_fields:
            side_entry = self.embedding_fields[sf]
            if int(side_entry.emb_size) != self.emb_size:
                self.side_proj[sf] = nn.Linear(int(side_entry.emb_size), self.emb_size, bias=False)

        # Learned attention weights (global, per source)
        self.attn_weights = nn.Parameter(torch.ones(self.num_sources) / self.num_sources)

        # MLP
        user_dim = sum(
            int(self.embedding_fields[fn].emb_size) for fn in self.plain_field_names
            if not (self.embedding_fields[fn].field_type == 0 and self.embedding_fields[fn].concat_type == "direct")
        )
        direct_dim = sum(
            int(self.embedding_fields[fn].dim) for fn in self.plain_field_names
            if self.embedding_fields[fn].field_type == 0 and self.embedding_fields[fn].concat_type == "direct"
        )
        mlp_input_dim = user_dim + direct_dim + self.emb_size

        mlp_hidden = list(eges_cfg.get("hidden_dims", [128, 64]))
        self.mlp = FullyConnectedLayer(
            input_dim=mlp_input_dim, hidden_dims=mlp_hidden,
            bias=[True] * len(mlp_hidden),
            batch_norm=bool(eges_cfg.get("batch_norm", True)),
            activation=str(eges_cfg.get("activation", "relu")),
            dropout=float(eges_cfg.get("dropout", 0.1)),
        )
        self.head = nn.Linear(mlp_hidden[-1], 1)

        self._validate_fields(model_cfg)
        self.reset_parameters()

    def _validate_fields(self, model_cfg: BaseModelConfig) -> None:
        if not model_cfg.embedding_fields:
            raise ValueError("embedding_fields must be a non-empty mapping")
        if "item_field" not in model_cfg.mlp:
            raise ValueError("EGES config must specify item_field")

    def reset_parameters(self) -> None:
        for emb in self.embedding_bags.values():
            nn.init.xavier_uniform_(emb.weight)
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, feature_bags: Mapping[str, Mapping[str, Tensor]]) -> Tensor:
        first_offsets = feature_bags[self.field_names[0]]["offsets"]
        batch_size = int(first_offsets.size(0))
        device = next(self.parameters()).device

        # --- Side info + item embeddings ---
        # EGES: attention-weighted average of item base + side info embeddings
        source_embs: list[Tensor] = []

        # 1. Base item embedding
        item_entry = self.embedding_fields[self.item_field]
        base_emb = embed_one_field(
            self.embedding_bags[str(item_entry.field_index)],
            feature_bags[self.item_field]["indices"],
            feature_bags[self.item_field]["offsets"],
            feature_bags[self.item_field]["weights"],
            device=device,
        )
        source_embs.append(base_emb)

        # 2. Side info embeddings (with optional dimension projection)
        for sf in self.side_fields:
            entry = self.embedding_fields[sf]
            side_emb = embed_one_field(
                self.embedding_bags[str(entry.field_index)],
                feature_bags[sf]["indices"],
                feature_bags[sf]["offsets"],
                feature_bags[sf]["weights"],
                device=device,
            )
            if sf in self.side_proj:
                side_emb = self.side_proj[sf](side_emb)
            source_embs.append(side_emb)

        # 3. Attention-weighted sum (softmax over sources)
        attn = F.softmax(self.attn_weights, dim=0)                         # [num_sources]
        stacked = torch.stack(source_embs, dim=1)                          # [B, num_sources, d]
        item_emb = (attn.unsqueeze(0).unsqueeze(-1) * stacked).sum(dim=1)  # [B, d]

        # --- User / plain features ---
        plain_embs: list[Tensor] = []
        for fn in self.plain_field_names:
            entry = self.embedding_fields[fn]
            if entry.field_type == 0 and entry.concat_type == "direct":
                plain_embs.append(feature_bags[fn]["weights"].view(-1, int(entry.dim)))
            else:
                plain_embs.append(embed_one_field(
                    self.embedding_bags[str(entry.field_index)],
                    feature_bags[fn]["indices"],
                    feature_bags[fn]["offsets"],
                    feature_bags[fn]["weights"],
                    device=device,
                ))

        user_feats = torch.cat(plain_embs, dim=-1) if plain_embs else torch.zeros(batch_size, 0, device=device)

        # --- Concat + MLP ---
        combined = torch.cat([user_feats, item_emb], dim=-1)
        hidden = self.mlp(combined)
        logit = self.head(hidden).squeeze(-1)
        return torch.sigmoid(logit)
