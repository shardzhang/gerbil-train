"""YouTube Deep Neural Network (YouTubeDNN) model for item recommendation.

Architecture (Covington et al., 2016):
    features → EmbeddingBag(mode=mean for behavior, sum for others)
            → MLP → user_embedding
            → Linear(target_size, bias=False) → logits

Key differences from GwEN:
    - Behavior fields use ``mode="mean"`` (sequence-length invariant)
    - Optional ``example_age_field`` with log(age+1) preprocessing
    - Linear head with ``bias=False`` (head.weight = item embedding matrix)
    - ``encode()`` returns user_embedding for ANN search in serving
"""

from __future__ import annotations

from typing import Mapping

import torch
from torch import Tensor, nn

from gerbil_train.config.model_config import YouTubeDNNModelConfig, FieldEntry
from gerbil_train.utils.embedding import embed_one_field
from gerbil_train.models.layers import FullyConnectedLayer
from gerbil_train.models.base_model import BaseModel

__all__ = ["YouTubeDNN"]


class YouTubeDNN(BaseModel):
    """YouTube Deep Neural Network for item recommendation (multi-class)."""

    def __init__(self, model_cfg: YouTubeDNNModelConfig) -> None:
        super().__init__()
        self._validate_fields(model_cfg)

        self.fields_cfg: Mapping[str, FieldEntry] = model_cfg.embedding_fields
        self.field_names = list(self.fields_cfg.keys())
        self.behavior_fields = set(model_cfg.behavior_fields)
        self.example_age_field = model_cfg.example_age_field
        self.head_bias = model_cfg.head_bias

        # Compute per-field embedding dimensions
        self.field_embedding_dims: dict[str, int] = {}
        for field_name, entry in self.fields_cfg.items():
            if field_name == self.example_age_field:
                self.field_embedding_dims[field_name] = 1
            elif entry.field_type == 1 or (entry.field_type == 0 and entry.concat_type == "emb"):
                self.field_embedding_dims[field_name] = int(entry.emb_size)
            elif entry.field_type == 0 and entry.concat_type == "direct":
                self.field_embedding_dims[field_name] = int(entry.dim)
            else:
                raise ValueError(f"Unsupported field_type={entry.field_type}")
        self.embedding_sum_dim = sum(self.field_embedding_dims.values())

        # EmbeddingBags for each field
        self.embedding_bags = nn.ModuleDict()
        for field_name, entry in self.fields_cfg.items():
            if field_name == self.example_age_field:
                continue
            is_behavior = field_name in self.behavior_fields
            cat_emb = entry.field_type == 1 or (entry.field_type == 0 and entry.concat_type == "emb")
            if cat_emb:
                key = str(entry.field_index)
                if key not in self.embedding_bags:
                    bag = nn.EmbeddingBag(
                        num_embeddings=int(entry.dim),
                        embedding_dim=int(entry.emb_size),
                        mode="mean" if is_behavior else "sum",
                    )
                    bag.field_name = f"{field_name}_{'mean' if is_behavior else 'sum'}"
                    self.embedding_bags[key] = bag
            elif entry.field_type == 0 and entry.concat_type == "direct":
                pass  # handled inline in forward

        # MLP
        mlp_cfg = model_cfg.mlp
        hidden_dims = list(mlp_cfg.get("hidden_dims", [256, 128]))
        self.input_bn = nn.BatchNorm1d(self.embedding_sum_dim) if mlp_cfg.get("input_batch_norm", False) else None
        self.mlp = FullyConnectedLayer(
            input_dim=self.embedding_sum_dim,
            hidden_dims=hidden_dims,
            bias=[True] * len(hidden_dims),
            batch_norm=bool(mlp_cfg.get("batch_norm", True)),
            activation=str(mlp_cfg.get("activation", "relu")),
            dropout=float(mlp_cfg.get("dropout", 0.0)),
        )
        final_hidden_dim = hidden_dims[-1] if hidden_dims else self.embedding_sum_dim
        # head.weight = item_embedding_matrix (no bias for ANN serving)
        self.head = nn.Linear(final_hidden_dim, int(model_cfg.target_size), bias=self.head_bias)
        self.reset_parameters()

    def _validate_fields(self, model_cfg: YouTubeDNNModelConfig) -> None:
        if not model_cfg.embedding_fields:
            raise ValueError("embedding_fields must be a non-empty mapping")
        if model_cfg.example_age_field and model_cfg.example_age_field not in model_cfg.embedding_fields:
            raise ValueError(f"example_age_field '{model_cfg.example_age_field}' not found in embedding_fields")

    def reset_parameters(self) -> None:
        for emb in self.embedding_bags.values():
            nn.init.xavier_uniform_(emb.weight)
        for module in self.mlp.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        nn.init.xavier_uniform_(self.head.weight)
        if self.head.bias is not None:
            nn.init.zeros_(self.head.bias)

    def encode(self, feature_bags: Mapping[str, Mapping[str, Tensor]]) -> Tensor:
        """Return user embedding ``[batch_size, final_hidden_dim]`` for ANN search."""
        first_offsets = feature_bags[self.field_names[0]]["offsets"]
        batch_size = int(first_offsets.size(0))
        device = next(self.parameters()).device

        field_embs: list[Tensor] = []
        for field_name, entry in self.fields_cfg.items():
            if field_name == self.example_age_field:
                ages = feature_bags[field_name]["weights"].view(-1, 1)
                field_embs.append((ages + 1).log())
                continue
            cat_emb = entry.field_type == 1 or (entry.field_type == 0 and entry.concat_type == "emb")
            if cat_emb:
                emb = embed_one_field(
                    self.embedding_bags[str(entry.field_index)],
                    feature_bags[field_name]["indices"],
                    feature_bags[field_name]["offsets"],
                    feature_bags[field_name]["weights"],
                    device=device,
                )
            elif entry.field_type == 0 and entry.concat_type == "direct":
                emb = feature_bags[field_name]["weights"].view(-1, int(entry.dim))
            else:
                raise ValueError(f"Unsupported field_type={entry.field_type}")
            field_embs.append(emb)

        concat = torch.cat(field_embs, dim=-1)
        if self.input_bn is not None:
            concat = self.input_bn(concat)
        return self.mlp(concat)

    def forward(self, feature_bags: Mapping[str, Mapping[str, Tensor]]) -> Tensor:
        """Forward pass returning logits over all target items.

        During training, pass logits to the loss function (CE/NCE/SampledSoftmax).
        During inference, use ``encode()`` for user embedding + ANN search.
        """
        user_emb = self.encode(feature_bags)
        logits = self.head(user_emb)
        return logits
