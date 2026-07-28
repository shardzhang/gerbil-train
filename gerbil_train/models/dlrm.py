"""DLRM (Deep Learning Recommendation Model) for CTR prediction.

DLRM processes sparse (categorical) features via embeddings and dense
features via a bottom MLP, then computes all pairwise dot products between
all feature embeddings. The interaction output is fed into a top MLP.

Reference: https://arxiv.org/abs/1906.00091 (RecSys 2019)
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

__all__ = ["DLRM"]


class DLRM(BaseModel):
    """Deep Learning Recommendation Model for CTR prediction."""

    def __init__(self, model_cfg: BaseModelConfig) -> None:
        super().__init__()
        self._validate_fields(model_cfg)

        self.fields_cfg: Mapping[str, FieldEntry] = model_cfg.embedding_fields
        self.field_names = list(self.fields_cfg.keys())

        # Separate sparse (categorical) and dense (continuous) fields
        self.sparse_fields = {
            n: e for n, e in self.fields_cfg.items()
            if not (e.field_type == 0 and e.concat_type == "direct")
        }
        self.dense_fields = {
            n: e for n, e in self.fields_cfg.items()
            if e.field_type == 0 and e.concat_type == "direct"
        }
        self.num_sparse = len(self.sparse_fields)
        self.num_dense = len(self.dense_fields)

        # Embedding bags for sparse features
        self.embedding_bags = nn.ModuleDict()
        for field_name, entry in self.sparse_fields.items():
            key = str(entry.field_index)
            if key not in self.embedding_bags:
                self.embedding_bags[key] = nn.EmbeddingBag(
                    num_embeddings=int(entry.dim),
                    embedding_dim=int(entry.emb_size),
                    mode="sum",
                )

        # Determine a uniform embedding dimension for interaction
        sparse_emb_sizes = {int(e.emb_size) for e in self.sparse_fields.values()}
        if sparse_emb_sizes:
            self.emb_size = max(sparse_emb_sizes)
        else:
            self.emb_size = 8

        # Project sparse embeddings to a uniform dimension if needed
        self.sparse_proj = nn.ModuleDict()
        for fn, entry in self.sparse_fields.items():
            if int(entry.emb_size) != self.emb_size:
                self.sparse_proj[fn] = nn.Linear(int(entry.emb_size), self.emb_size, bias=False)

        # Bottom MLP for dense features (process concat'd dense features → d-dim vector)
        dlrm_cfg: dict[str, Any] = model_cfg.mlp
        dense_input_dim = sum(int(e.dim) for e in self.dense_fields.values())
        bottom_hidden = list(dlrm_cfg.get("bottom_hidden", [128, 64]))
        if dense_input_dim > 0 and bottom_hidden:
            self.bottom_mlp = FullyConnectedLayer(
                input_dim=dense_input_dim, hidden_dims=bottom_hidden,
                bias=[True] * len(bottom_hidden),
                batch_norm=True, activation="relu", dropout=0.1,
            )
            self.dense_emb_dim = bottom_hidden[-1]
            # Also project dense to interaction dimension if needed
            if self.dense_emb_dim != self.emb_size:
                self.dense_proj = nn.Linear(self.dense_emb_dim, self.emb_size, bias=False)
            else:
                self.dense_proj = nn.Identity()
        else:
            self.bottom_mlp = None
            self.dense_proj = None

        # Total number of features for interaction (sparse + 1 dense embedding)
        self.num_features = self.num_sparse + (1 if self.bottom_mlp is not None else 0)
        num_pairs = self.num_features * (self.num_features - 1) // 2

        # Top MLP
        top_input_dim = num_pairs + (self.emb_size if self.bottom_mlp is not None else 0)
        top_hidden = list(dlrm_cfg.get("top_hidden", [256, 128, 64]))
        self.top_mlp = FullyConnectedLayer(
            input_dim=top_input_dim, hidden_dims=top_hidden,
            bias=[True] * len(top_hidden),
            batch_norm=True, activation="relu", dropout=0.1,
        )
        self.head = nn.Linear(top_hidden[-1], 1)
        self.reset_parameters()

    def _validate_fields(self, model_cfg: BaseModelConfig) -> None:
        if not model_cfg.embedding_fields:
            raise ValueError("embedding_fields must be a non-empty mapping")

    def reset_parameters(self) -> None:
        for emb in self.embedding_bags.values():
            nn.init.xavier_uniform_(emb.weight)
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, feature_bags: Mapping[str, Mapping[str, Tensor]]) -> Tensor:
        first_offsets = feature_bags[self.field_names[0]]["offsets"]
        batch_size = int(first_offsets.size(0))
        device = next(self.parameters()).device

        # Collect feature embeddings (all projected to emb_size)
        feature_embs: list[Tensor] = []

        # Sparse features
        for fn, entry in self.sparse_fields.items():
            emb = embed_one_field(
                self.embedding_bags[str(entry.field_index)],
                feature_bags[fn]["indices"],
                feature_bags[fn]["offsets"],
                feature_bags[fn]["weights"],
                device=device,
            )
            if fn in self.sparse_proj:
                emb = self.sparse_proj[fn](emb)
            feature_embs.append(emb)

        # Dense features → bottom MLP → one dense embedding
        if self.bottom_mlp is not None and self.dense_fields:
            dense_list: list[Tensor] = []
            for fn, entry in self.dense_fields.items():
                dense_list.append(feature_bags[fn]["weights"].view(-1, int(entry.dim)))
            dense_input = torch.cat(dense_list, dim=-1)
            dense_emb = self.dense_proj(self.bottom_mlp(dense_input))
            feature_embs.append(dense_emb)

        # Stack all features: [B, N, d]
        if not feature_embs:
            return torch.zeros(batch_size, device=device)
        x = torch.stack(feature_embs, dim=1)
        N = x.size(1)

        # All pairwise dot products (upper triangle, excluding diagonal)
        # Gram matrix: [B, N, N]
        gram = torch.bmm(x, x.transpose(1, 2))
        # Extract upper triangle (i < j) → [B, N*(N-1)/2]
        triu_indices = torch.triu_indices(N, N, offset=1, device=device)
        interaction = gram[:, triu_indices[0], triu_indices[1]]

        # Concat with dense embedding (optional, per original DLRM)
        if self.bottom_mlp is not None:
            combined = torch.cat([interaction, dense_emb], dim=-1)
        else:
            combined = interaction

        # Top MLP → sigmoid
        hidden = self.top_mlp(combined)
        logit = self.head(hidden).squeeze(-1)
        return torch.sigmoid(logit)
