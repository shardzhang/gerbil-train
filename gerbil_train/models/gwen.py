"""Shared GwEN base model implementation."""

from __future__ import annotations

from typing import Mapping

import torch
from torch import Tensor, nn

from gerbil_train.config.model_config import BaseModelConfig
from gerbil_train.utils.embedding import embed_one_field
from gerbil_train.models.layers import FullyConnectedLayer
from gerbil_train.config.model_config import FieldEntry
from gerbil_train.models.base_model import BaseModel

__all__ = ["GwENBaseModel", "GwENBinaryModel", "GwENMulticlassModel"]


class GwENBaseModel(BaseModel):
    """Shared GwEN implementation parameterized by ``task``. Use ``GwENBinary`` or ``GwENMulticlass``."""

    def __init__(self, config: BaseModelConfig, task: str = "multiclass") -> None:
        super().__init__()
        self.fields_cfg: Mapping[str, FieldEntry] = config.embedding_fields
        self._validate_fields(config)

        self.task = task
        # dict[field_name, dim]
        self.field_embedding_dims: dict[str, int] = {}
        # dict[str(field_index), nn.EmbeddingBag], 支持词表共享
        self.field_embeddings: nn.ModuleDict = nn.ModuleDict()
        for field_name, entry in self.fields_cfg.items():
            vocab_size = int(entry.dim)
            embedding_size = int(entry.emb_size)
            self.field_embedding_dims[field_name] = embedding_size
            key = str(entry.field_index)
            if key in self.field_embeddings:
                print(f"Field {field_name} (field_index={entry.field_index})共享词表")
                continue
            bag = nn.EmbeddingBag(
                num_embeddings=vocab_size,
                embedding_dim=embedding_size,
                mode="sum",
                include_last_offset=False,
            )
            bag.field_name = field_name
            self.field_embeddings[key] = bag

        self.enable_attention = bool(config.field_attention.get("enabled", False))
        if self.enable_attention:
            self.field_attention = nn.ModuleDict({
                field_name: nn.Linear(self.field_embedding_dims[field_name], 1, bias=False)
                for field_name in self.fields_cfg.keys()
            })

        self.embedding_sum_dim = sum(self.field_embedding_dims.values())
        mlp_cfg = config.mlp
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

    def _validate_fields(self, model_cfg: BaseModelConfig) -> None:
        if not model_cfg.embedding_fields:
            raise ValueError("embedding_fields must be a non-empty mapping")

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


    def _embed_one_field(self, field_index: int, indices: Tensor, offsets: Tensor,
                         weights: Tensor, *, batch_size: int, device: torch.device) -> Tensor:
        return embed_one_field(self.field_embeddings[str(field_index)], indices, offsets, weights, device=device)


    def encode(self, feature_bags: Mapping[int, Mapping[str, Tensor]]) -> Tensor:
        """Encode input features into dense representations.

        :return: Feature tensor of shape ``[batch_size, final_hidden_dim]``
        """
        if not isinstance(feature_bags, Mapping) or not feature_bags:
            raise ValueError("feature_bags must be a non-empty mapping")

        first_field_name = next(iter(self.fields_cfg.keys()))
        first_offsets = feature_bags[first_field_name]["offsets"]
        batch_size = int(first_offsets.size(0))
        device = next(self.parameters()).device

        field_embeddings: dict[int, Tensor] = {}
        for field_name, cfg in self.fields_cfg.items():
            if field_name not in feature_bags:
                raise ValueError(f"Missing feature bag for field {field_name}")
            if not isinstance(feature_bags[field_name], Mapping):
                raise ValueError(f"feature_bags[{field_name}] must be a mapping")
            field_embeddings[field_name] = self._embed_one_field(
                cfg.field_index,
                feature_bags[field_name]["indices"],
                feature_bags[field_name]["offsets"],
                feature_bags[field_name]["weights"],
                batch_size=batch_size,
                device=device,
            )

        if self.enable_attention:
            field_scores = torch.cat(
                [self.field_attention[fn](field_embeddings[fn]) for fn in self.fields_cfg.keys()], dim=-1,
            )
            field_weights = torch.softmax(field_scores, dim=-1)
            weighted_embeddings = [
                field_embeddings[fn] * field_weights[:, i].unsqueeze(-1) for i, fn in enumerate(self.fields_cfg.keys())
            ]
            input_emb = torch.cat(weighted_embeddings, dim=-1)
        else:
            input_emb = torch.cat([field_embeddings[fn] for fn in self.fields_cfg.keys()], dim=-1)

        # [batch, embedding_sum_dim]
        if self.input_bn is not None:
            input_emb = self.input_bn(input_emb)
        
        # [batch, final_hidden_dim]
        hidden = self.mlp(input_emb)
        return hidden


    def forward(self, feature_bags: Mapping[str, Mapping[str, Tensor]]) -> Tensor:
        """Forward pass of the model.
        :param feature_bags: Feature bags for each field.
        :return: sigmoid of logits for binary classification or logits for multi-class classification.
        """
        logits = self.head(self.encode(feature_bags))
        if self.task == "binary":
            return torch.sigmoid(logits).squeeze(-1)
        if self.task == "multiclass":
            return logits
        raise ValueError(f"Unsupported task: {self.task}")


class GwENBinaryModel(GwENBaseModel):
    """GwEN for binary classification."""
    def __init__(self, config: BaseModelConfig) -> None:
        super().__init__(config, task="binary")


class GwENMulticlassModel(GwENBaseModel):
    """GwEN for multi-class classification."""
    def __init__(self, config: BaseModelConfig) -> None:
        super().__init__(config, task="multiclass")
