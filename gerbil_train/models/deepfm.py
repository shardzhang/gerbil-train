"""DeepFM model with EmbeddingBag support for TFRecord input."""

from __future__ import annotations

from typing import Mapping

import torch
from torch import Tensor, nn

from gerbil_train.config import DeepFMConfig
from gerbil_train.utils.embedding import embed_one_field
from gerbil_train.utils.nn import build_mlp

__all__ = ["DeepFM"]


class DeepFM(nn.Module):
    """DeepFM model for recommendation and CTR prediction."""

    def __init__(self, config: DeepFMConfig) -> None:
        super().__init__()

        self.field_names = list(config.field_names)
        if not self.field_names:
            raise ValueError("DeepFM requires at least one field.")

        self.output_activation = str(config.output.get("activation", "none"))

        self.linear_embeddings = nn.ModuleDict()
        self.feature_embeddings = nn.ModuleDict()
        self.embedding_dim = int(config.embedding_dim)

        for field_name in self.field_names:
            if config.embedding_fields and field_name in config.embedding_fields:
                entry = config.embedding_fields[field_name]
                vocab_size = int(entry.vocab_size)
            elif config.sparse_fields and field_name in config.sparse_fields:
                vocab_size = int(config.sparse_fields[field_name].vocab_size)
            else:
                raise ValueError(f"Field {field_name} not found in config")

            self.linear_embeddings[field_name] = nn.EmbeddingBag(
                num_embeddings=vocab_size, embedding_dim=1, mode="sum",
            )
            self.feature_embeddings[field_name] = nn.EmbeddingBag(
                num_embeddings=vocab_size, embedding_dim=self.embedding_dim, mode="sum",
            )

        deep_cfg = config.deep
        self.deep_hidden_dims = list(deep_cfg.get("hidden_dims", [128, 64]))
        self.embedding_sum_dim = len(self.field_names) * self.embedding_dim
        self.deep_network = build_mlp(
            input_dim=self.embedding_sum_dim,
            hidden_dims=self.deep_hidden_dims,
            batch_norm=bool(deep_cfg.get("batch_norm", False)),
            activation=str(deep_cfg.get("activation", "relu")),
            dropout=float(deep_cfg.get("dropout", 0.0)),
        )
        deep_output_dim = self.deep_hidden_dims[-1] if self.deep_hidden_dims else self.embedding_sum_dim
        self.deep_head = nn.Linear(deep_output_dim, 1)
        self.bias = nn.Parameter(torch.zeros(1))
        self.reset_parameters()

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
        first_offsets = feature_bags[self.field_names[0]]["offsets"]
        batch_size = int(first_offsets.size(0))
        device = next(self.parameters()).device

        logits = self.bias.expand(batch_size).to(device)
        feature_emb_list: list[Tensor] = []

        for fn in self.field_names:
            bag = feature_bags[fn]
            linear_emb = embed_one_field(
                self.linear_embeddings[fn], bag["indices"], bag["offsets"],
                bag["weights"], device=device,
            )
            logits = logits + linear_emb.squeeze(-1)

            feature_emb = embed_one_field(
                self.feature_embeddings[fn], bag["indices"], bag["offsets"],
                bag["weights"], device=device,
            )
            feature_emb_list.append(feature_emb)

        # FM second-order term
        stacked = torch.stack(feature_emb_list, dim=1)
        summed = stacked.sum(dim=1)
        sum_of_squares = (stacked * stacked).sum(dim=1)
        logits = logits + 0.5 * (summed * summed - sum_of_squares).sum(dim=1)

        # Deep term
        deep_input = torch.cat(feature_emb_list, dim=-1)
        logits = logits + self.deep_head(self.deep_network(deep_input)).squeeze(-1)

        if self.output_activation == "sigmoid":
            return torch.sigmoid(logits)
        return logits
