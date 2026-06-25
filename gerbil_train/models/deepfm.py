"""DeepFM model with EmbeddingBag support for TFRecord input."""

from __future__ import annotations

from typing import Mapping

import torch
from torch import Tensor, nn

from gerbil_train.config.model_config import DeepFMModelConfig, FieldEntry
from gerbil_train.utils.embedding import embed_one_field
from gerbil_train.models.layers import FullyConnectedLayer
from gerbil_train.models.base_model import BaseModel

__all__ = ["DeepFM"]


class DeepFM(BaseModel):
    """DeepFM model for recommendation and CTR prediction."""

    def __init__(self, model_cfg: DeepFMModelConfig) -> None:
        super().__init__()
        self.validate_fields(model_cfg)

        self.fields_cfg: Mapping[str, FieldEntry] = model_cfg.embedding_fields
        self.field_names = list(self.fields_cfg.keys())
        self.embedding_sum_dim = sum(entry.emb_size for entry in self.fields_cfg.values())

        self.linear_embeddings = nn.ModuleDict()
        self.feature_embeddings = nn.ModuleDict()
        for field_name, entry in self.fields_cfg.items():
            self.linear_embeddings[str(entry.field_index)] = nn.EmbeddingBag(
                num_embeddings=entry.dim, 
                embedding_dim=1, 
                mode="mean",
            )
            self.feature_embeddings[str(entry.field_index)] = nn.EmbeddingBag(
                num_embeddings=entry.dim,
                embedding_dim=entry.emb_size,
                mode="mean",
            )

        mlp_cfg = model_cfg.mlp
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


    def validate_fields(self, model_cfg: DeepFMModelConfig) -> None:
        if not model_cfg.embedding_fields:
            raise ValueError("embedding_fields must be a non-empty mapping")
        first_emb = next(iter(model_cfg.embedding_fields.values())).emb_size
        if not all(entry.emb_size == first_emb for entry in model_cfg.embedding_fields.values()):
            raise ValueError("All fields must have the same embedding size")


    def reset_parameters(self) -> None:
        """Reset model parameters."""
        nn.init.zeros_(self.bias)
        if self.deep_head is not None:
            nn.init.xavier_uniform_(self.deep_head.weight)
            nn.init.zeros_(self.deep_head.bias)
        for emb in self.linear_embeddings.values():
            nn.init.xavier_uniform_(emb.weight)
        for emb in self.feature_embeddings.values():
            nn.init.xavier_uniform_(emb.weight)


    def forward(self, feature_bags: Mapping[str, Mapping[str, Tensor]]) -> Tensor:
        """Forward pass of the DeepFM model."""
        
        # 1. Linear term
        first_offsets = feature_bags[self.field_names[0]]["offsets"]
        batch_size = int(first_offsets.size(0))
        device = next(self.parameters()).device

        # [batch_size, ]
        logits = self.bias.expand(batch_size).to(device)
        feature_emb_list: list[Tensor] = []

        for field_name, entry in self.fields_cfg.items():
            # [batch_size, 1]
            linear_emb = embed_one_field(
                self.linear_embeddings[str(entry.field_index)], 
                feature_bags[field_name]["indices"], 
                feature_bags[field_name]["offsets"],
                feature_bags[field_name]["weights"], 
                device=device,
            )
            # [batch_size, ]
            logits = logits + linear_emb.squeeze(-1)

            # [batch_size, embedding_dim]
            feature_emb = embed_one_field(
                self.feature_embeddings[str(entry.field_index)], 
                feature_bags[field_name]["indices"], 
                feature_bags[field_name]["offsets"],
                feature_bags[field_name]["weights"], 
                device=device,
            )
            feature_emb_list.append(feature_emb)

        # 2. FM second-order term
        # FM 公式: 0.5 * Σ((Σ f)^2 - Σ(f^2))
        # $$\text{FM} = \frac{1}{2}\sum_{f=1}^{k}\left((\sum_{i=1}^{n} v_{i,f})^2 - \sum_{i=1}^{n} v_{i,f}^2\right)$$
        # $$\text{FM} = \sum_{i=1}^{n}\sum_{j=i+1}^{n} \langle v_i, v_j \rangle$$
        # [batch_size, num_fields, embedding_dim]
        stacked = torch.stack(feature_emb_list, dim=1)
        # [batch_size, embedding_dim]
        summed = stacked.sum(dim=1)
        # [batch_size, embedding_dim]
        sum_of_squares = (stacked * stacked).sum(dim=1)
        # [batch_size, ]
        logits = logits + 0.5 * (summed * summed - sum_of_squares).sum(dim=1)

        # 3. Deep term
        # [batch_size, num_fields * embedding_dim]
        deep_input = torch.cat(feature_emb_list, dim=-1)
        # [batch_size, 1]
        logits = logits + self.deep_head(self.deep_network(deep_input)).squeeze(-1)

        return torch.sigmoid(logits)