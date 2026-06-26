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
        self._validate_fields(model_cfg)

        self.fields_cfg: Mapping[str, FieldEntry] = model_cfg.embedding_fields
        self.field_names = list(self.fields_cfg.keys())
        self.embedding_sum_dim = sum(entry.emb_size for entry in self.fields_cfg.values())
        self.num_fields = len(self.field_names)

        self.linear_embeddings = nn.ModuleDict()
        self.feature_embedding_bags = nn.ModuleDict()
        for _, entry in self.fields_cfg.items():
            key = str(entry.field_index)
            # 1. Linear embedding: vocab → 1, used for the 1st-order term
            linear_bag = nn.EmbeddingBag(
                num_embeddings=entry.dim,
                embedding_dim=1,
                mode="sum",
            )
            linear_bag.field_name = entry.field_name + "_fm"
            self.linear_embeddings[key] = linear_bag
        
            # 2. Feature embedding: vocab → k, shared by FM 2nd-order and Deep terms
            feature_bag = nn.EmbeddingBag(
                num_embeddings=entry.dim,
                embedding_dim=entry.emb_size,
                mode="sum",
            )
            feature_bag.field_name = entry.field_name + "_deep"
            self.feature_embedding_bags[key] = feature_bag

        mlp_cfg = model_cfg.mlp
        # BatchNorm on concatenated feature embeddings to prevent logit saturation from mode="sum"
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


    def _validate_fields(self, model_cfg: DeepFMModelConfig) -> None:
        """Validate that the embedding fields are properly configured."""
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
        for emb in self.feature_embedding_bags.values():
            nn.init.xavier_uniform_(emb.weight)


    def forward(self, feature_bags: Mapping[str, Mapping[str, Tensor]]) -> Tensor:
        """Forward pass of the DeepFM model.

        DeepFM = Linear (1st-order) + FM (2nd-order pair-wise) + Deep (high-order non-linear),
        all sharing the same feature embeddings.
        """
        first_offsets = feature_bags[self.field_names[0]]["offsets"]
        batch_size = int(first_offsets.size(0))
        device = next(self.parameters()).device

        # 1. Linear term (1st-order): sum of per-field linear embeddings + global bias
        logits = self.bias.expand(batch_size).to(device)
        linear_sum = torch.zeros(batch_size, device=device)
        feature_emb_list: list[Tensor] = []

        for field_name, entry in self.fields_cfg.items():
            key = str(entry.field_index)
            linear_emb = embed_one_field(
                self.linear_embeddings[key],
                feature_bags[field_name]["indices"],
                feature_bags[field_name]["offsets"],
                feature_bags[field_name]["weights"],
                device=device,
            )
            linear_sum = linear_sum + linear_emb.squeeze(-1)

            feature_emb = embed_one_field(
                self.feature_embedding_bags[key],
                feature_bags[field_name]["indices"],
                feature_bags[field_name]["offsets"],
                feature_bags[field_name]["weights"],
                device=device,
            )
            feature_emb_list.append(feature_emb)

        # Concatenate and normalize feature embeddings before FM/Deep to prevent logit saturation
        # [batch_size, num_fields * embedding_dim]
        concat = torch.cat(feature_emb_list, dim=-1)
        if self.input_bn is not None:
            concat = self.input_bn(concat)
        
        # [batch_size, num_fields, embedding_dim]
        feature_embs = concat.view(batch_size, self.num_fields, -1)

        # 2. FM second-order term: pair-wise feature interactions
        stacked = feature_embs                                                          # [B, n, k]
        summed = stacked.sum(dim=1)                                                     # [B, k]
        sum_of_squares = (stacked * stacked).sum(dim=1)                                 # [B, k]
        logits = logits + linear_sum / self.num_fields + 0.5 * (summed * summed - sum_of_squares).sum(dim=1) / self.num_fields

        # 3. Deep term: high-order non-linear interactions via MLP
        deep_input = concat                                                             # [B, n*k]
        logits = logits + self.deep_head(self.deep_network(deep_input)).squeeze(-1)     # [B, ]

        return torch.sigmoid(logits)