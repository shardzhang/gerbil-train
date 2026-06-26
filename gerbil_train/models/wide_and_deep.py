"""Wide & Deep model for CTR prediction.

Wide & Deep = Wide (1st-order linear) + Deep (MLP),
sharing the same feature embeddings.
"""

from __future__ import annotations

from typing import Mapping

import torch
from torch import Tensor, nn

from gerbil_train.config.model_config import WideAndDeepModelConfig, FieldEntry
from gerbil_train.utils.embedding import embed_one_field
from gerbil_train.models.layers import FullyConnectedLayer
from gerbil_train.models.base_model import BaseModel

__all__ = ["WideAndDeep"]


class WideAndDeep(BaseModel):
    """Wide & Deep model for recommendation and CTR prediction."""

    def __init__(self, model_cfg: WideAndDeepModelConfig) -> None:
        super().__init__()
        self._validate_fields(model_cfg)

        self.fields_cfg: Mapping[str, FieldEntry] = model_cfg.embedding_fields
        self.field_names = list(self.fields_cfg.keys())
        self.num_fields = len(self.field_names)

        # Compute per-field embedding dimensions
        self.field_embedding_dims: dict[str, int] = {}
        for field_name, entry in self.fields_cfg.items():
            is_cat = entry.field_type == 1
            is_emb = entry.field_type == 0 and entry.concat_type == "emb"
            is_direct = entry.field_type == 0 and entry.concat_type == "direct"
            if is_cat or is_emb:
                self.field_embedding_dims[field_name] = int(entry.emb_size)
            elif is_direct:
                self.field_embedding_dims[field_name] = int(entry.dim)
            else:
                raise ValueError(
                    f"Unsupported field_type={entry.field_type} "
                    f"concat_type={entry.concat_type} for {field_name}"
                )
        self.embedding_sum_dim = sum(self.field_embedding_dims.values())

        # Linear (wide) embeddings: vocab → 1, 1st-order term
        self.linear_embeddings = nn.ModuleDict()
        # Feature (deep) embeddings: vocab → k, shared by deep term
        self.feature_embeddings = nn.ModuleDict()
        for field_name, entry in self.fields_cfg.items():
            if entry.field_type == 0 and entry.concat_type == "direct":
                continue  # direct concat fields skip embedding
            key = str(entry.field_index)
            # Linear embedding: vocab → 1, for wide term
            if key not in self.linear_embeddings:
                self.linear_embeddings[key] = nn.EmbeddingBag(
                    num_embeddings=int(entry.dim),
                    embedding_dim=1,
                    mode="sum",
                )
            # Feature embedding: vocab → k, for deep term
            if key not in self.feature_embeddings:
                self.feature_embeddings[key] = nn.EmbeddingBag(
                    num_embeddings=int(entry.dim),
                    embedding_dim=int(entry.emb_size),
                    mode="sum",
                )

        # Deep network
        mlp_cfg = model_cfg.mlp
        # BatchNorm on concatenated feature embeddings
        self.input_bn = nn.BatchNorm1d(self.embedding_sum_dim) if mlp_cfg.get("input_batch_norm", False) else None

        hidden_dims = list(mlp_cfg.get("hidden_dims", [128, 64]))
        self.deep_network = FullyConnectedLayer(
            input_dim=self.embedding_sum_dim,
            hidden_dims=hidden_dims,
            bias=[True] * len(hidden_dims),
            batch_norm=bool(mlp_cfg.get("batch_norm", False)),
            activation=str(mlp_cfg.get("activation", "relu")),
            dropout=float(mlp_cfg.get("dropout", 0.0)),
        )
        deep_output_dim = hidden_dims[-1] if hidden_dims else self.embedding_sum_dim
        self.deep_head = nn.Linear(deep_output_dim, 1)
        self.bias = nn.Parameter(torch.zeros(1))
        self.reset_parameters()

    @staticmethod
    def _validate_fields(model_cfg: WideAndDeepModelConfig) -> None:
        if not model_cfg.embedding_fields:
            raise ValueError("embedding_fields must be a non-empty mapping")

    def reset_parameters(self) -> None:
        nn.init.zeros_(self.bias)
        if self.deep_head is not None:
            nn.init.xavier_uniform_(self.deep_head.weight)
            nn.init.zeros_(self.deep_head.bias)
        for emb in self.linear_embeddings.values():
            nn.init.xavier_uniform_(emb.weight)
        for emb in self.feature_embeddings.values():
            nn.init.xavier_uniform_(emb.weight)

    def forward(self, feature_bags: Mapping[str, Mapping[str, Tensor]]) -> Tensor:
        """Forward pass of the Wide & Deep model.

        Wide & Deep = Wide (1st-order linear) + Deep (MLP),
        sharing the same feature embeddings.

        $$ \text{W&D} = \text{sigmoid}(
            w_0 + \sum_{i} w_i x_i
            + \text{MLP}(\text{concat}(\mathbf{e}_1, ..., \mathbf{e}_n))
        ) $$
        """
        first_offsets = feature_bags[self.field_names[0]]["offsets"]
        batch_size = int(first_offsets.size(0))
        device = next(self.parameters()).device

        # ──────────────────────────────────────────────
        # 1. Wide term (1st-order linear)
        #    w_0 + Σ w_i · x_i
        # ──────────────────────────────────────────────
        linear_sum = torch.zeros(batch_size, device=device)
        feature_emb_list: list[Tensor] = []

        for field_name, entry in self.fields_cfg.items():
            if entry.field_type == 0 and entry.concat_type == "direct":
                # Direct concat: skip linear, use raw values for deep only
                feature_emb_list.append(
                    feature_bags[field_name]["weights"].view(-1, int(entry.dim))
                )
                continue

            key = str(entry.field_index)
            # Linear embedding: vocab → 1, scalar per field
            linear_emb = embed_one_field(
                self.linear_embeddings[key],
                feature_bags[field_name]["indices"],
                feature_bags[field_name]["offsets"],
                feature_bags[field_name]["weights"],
                device=device,
            )
            linear_sum = linear_sum + linear_emb.squeeze(-1)

            # Feature embedding: vocab → k, for deep term
            feature_emb = embed_one_field(
                self.feature_embeddings[key],
                feature_bags[field_name]["indices"],
                feature_bags[field_name]["offsets"],
                feature_bags[field_name]["weights"],
                device=device,
            )
            feature_emb_list.append(feature_emb)

        # ──────────────────────────────────────────────
        # 2. Deep term: high-order non-linear interactions via MLP
        #    Deep = MLP(concat(\mathbf{e}_1, ..., \mathbf{e}_n))
        # ──────────────────────────────────────────────
        # [batch_size, num_fields * embedding_dim]
        deep_input = torch.cat(feature_emb_list, dim=-1)
        if self.input_bn is not None:
            deep_input = self.input_bn(deep_input)
        # [batch_size, ]
        deep_logit = self.deep_head(self.deep_network(deep_input)).squeeze(-1)

        # Total logits = wide + deep
        logits = linear_sum + self.bias + deep_logit
        return torch.sigmoid(logits)
