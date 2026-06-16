"""Shared GwEN base model implementation."""

from __future__ import annotations

from typing import Mapping

import torch
from torch import Tensor, nn

from gerbil_train.config import GwENModelConfig
from gerbil_train.utils.embedding import embed_one_field, to_device
from gerbil_train.utils.nn import build_mlp

__all__ = ["GwENBase", "GwENBinary", "GwENMulticlass"]


class GwENBase(nn.Module):
    """Shared GwEN implementation parameterized by ``task``. Use ``GwENBinary`` or ``GwENMulticlass``."""

    def __init__(self, config: GwENModelConfig, task: str = "multiclass") -> None:
        super().__init__()
        fields_cfg = config.embedding_fields
        if not fields_cfg:
            raise ValueError("embedding_fields must be a non-empty mapping")

        self.task = task
        self.field_names = list(fields_cfg.keys())
        self.field_embedding_dims: dict[str, int] = {}
        self.field_embeddings = nn.ModuleDict()
        for field_name in self.field_names:
            entry = fields_cfg[field_name]
            vocab_size = int(entry.vocab_size)
            embedding_dim = int(entry.emb_dim)
            self.field_embedding_dims[field_name] = embedding_dim
            self.field_embeddings[field_name] = nn.EmbeddingBag(
                num_embeddings=vocab_size, embedding_dim=embedding_dim,
                mode="sum", include_last_offset=False,
            )

        self.enable_attention = bool(config.attention.get("enabled", False))
        if self.enable_attention:
            self.field_attention = nn.ModuleDict({
                field_name: nn.Linear(self.field_embedding_dims[field_name], 1, bias=False)
                for field_name in self.field_names
            })

        self.embedding_sum_dim = sum(self.field_embedding_dims.values())
        mlp_cfg = config.mlp
        hidden_dims = list(mlp_cfg.get("hidden_dims", [256, 128]))
        self.input_bn = nn.BatchNorm1d(self.embedding_sum_dim) if mlp_cfg.get("input_batch_norm", False) else None
        self.mlp = build_mlp(
            input_dim=self.embedding_sum_dim, hidden_dims=hidden_dims,
            batch_norm=bool(mlp_cfg.get("batch_norm", True)),
            activation=str(mlp_cfg.get("activation", "relu")),
            dropout=float(mlp_cfg.get("dropout", 0.0)),
        )
        final_hidden_dim = hidden_dims[-1] if hidden_dims else self.embedding_sum_dim

        if self.task == "binary":
            self.head = nn.Linear(final_hidden_dim, 1)
        elif self.task == "multiclass":
            self.target_size = int(config.target_size)
            if self.target_size <= 0:
                raise ValueError("target_size must be positive")
            self.head = nn.Linear(final_hidden_dim, self.target_size)
        else:
            raise ValueError(f"Unsupported task: {task}")

        self.reset_parameters()

    def reset_parameters(self) -> None:
        for embedding in self.field_embeddings.values():
            nn.init.xavier_uniform_(embedding.weight)
        if self.enable_attention:
            for linear in self.field_attention.values():
                nn.init.xavier_uniform_(linear.weight)
        for module in self.mlp.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def _embed_one_field(self, field_name: str, indices: Tensor, offsets: Tensor,
                         weights: Tensor, *, batch_size: int, device: torch.device) -> Tensor:
        return embed_one_field(self.field_embeddings[field_name], indices, offsets, weights, device=device)

    def encode(self, feature_bags: Mapping[str, Mapping[str, Tensor]]) -> Tensor:
        """Encode input features into dense representations.

        :return: Feature tensor of shape ``[batch_size, final_hidden_dim]``
        """
        if not isinstance(feature_bags, Mapping) or not feature_bags:
            raise ValueError("feature_bags must be a non-empty mapping")

        first_field_name = self.field_names[0]
        first_offsets = feature_bags[first_field_name]["offsets"]
        batch_size = int(first_offsets.size(0))
        device = next(self.parameters()).device

        field_embeddings: dict[str, Tensor] = {}
        for field_name in self.field_names:
            if field_name not in feature_bags:
                raise ValueError(f"Missing feature bag for field {field_name}")
            bag = feature_bags[field_name]
            if not isinstance(bag, Mapping):
                raise ValueError(f"feature_bags[{field_name}] must be a mapping")
            field_embeddings[field_name] = self._embed_one_field(
                field_name, bag["indices"], bag["offsets"], bag["weights"],
                batch_size=batch_size, device=device,
            )

        if self.enable_attention:
            field_scores = torch.cat(
                [self.field_attention[fn](field_embeddings[fn]) for fn in self.field_names], dim=-1,
            )
            field_weights = torch.softmax(field_scores, dim=-1)
            weighted_embeddings = [
                field_embeddings[fn] * field_weights[:, i].unsqueeze(-1) for i, fn in enumerate(self.field_names)
            ]
            concat_embedding = torch.cat(weighted_embeddings, dim=-1)
        else:
            concat_embedding = torch.cat([field_embeddings[fn] for fn in self.field_names], dim=-1)

        if self.input_bn is not None:
            concat_embedding = self.input_bn(concat_embedding)

        return self.mlp(concat_embedding)

    def forward(self, feature_bags: Mapping[str, Mapping[str, Tensor]]) -> Tensor:
        hidden = self.encode(feature_bags)
        if self.task == "binary":
            return torch.sigmoid(self.head(hidden)).squeeze(-1)
        return self.head(hidden)


class GwENBinary(GwENBase):
    """GwEN for binary classification."""

    def __init__(self, config: GwENModelConfig) -> None:
        super().__init__(config, task="binary")


class GwENMulticlass(GwENBase):
    """GwEN for multi-class classification."""

    def __init__(self, config: GwENModelConfig) -> None:
        super().__init__(config, task="multiclass")

