"""GwEN (Group-wise Embedding Network) model.

GwEN follows an Embedding + MLP design for multi-class recommendation:
    1. Field-wise sparse features -> per-field embedding bags
    2. Optional field-wise attention reweighting
    3. Concatenate all field embeddings
    4. MLP head -> multi-class logits
"""
from __future__ import annotations
from typing import Any, Mapping

import torch
from torch import Tensor, nn

from gerbil_train.config import GwENModelConfig
from gerbil_train.utils.nn import build_mlp

__all__ = ["GwEN"]


class GwEN(nn.Module):
    """Group-wise Embedding Network for multi-class target prediction."""

    def __init__(self, config: GwENModelConfig) -> None:
        """Initialize a GwEN model from config.

        :param config: GwEN model configuration
        """
        super().__init__()

        fields_cfg = config.embedding_fields
        if not fields_cfg:
            raise ValueError("embedding_fields must be a non-empty mapping")

        # Ordered list of field names, used throughout the forward pass
        self.field_names = list(fields_cfg.keys())
        self.target_size = int(config.target_size)
        if self.target_size <= 0:
            raise ValueError("target_size must be positive")

        self.field_embedding_dims: dict[str, int] = {}
        self.field_embeddings = nn.ModuleDict()
        for field_name in self.field_names:
            entry = fields_cfg[field_name]
            vocab_size = int(entry.vocab_size)
            embedding_dim = int(entry.emb_dim)
            self.field_embedding_dims[field_name] = embedding_dim
            self.field_embeddings[field_name] = nn.EmbeddingBag(
                num_embeddings=vocab_size,
                embedding_dim=embedding_dim,
                mode="sum",
                include_last_offset=False,
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
            input_dim=self.embedding_sum_dim,
            hidden_dims=hidden_dims,
            batch_norm=bool(mlp_cfg.get("batch_norm", True)),
            activation=str(mlp_cfg.get("activation", "relu")),
            dropout=float(mlp_cfg.get("dropout", 0.0)),
        )

        final_hidden_dim = hidden_dims[-1] if hidden_dims else self.embedding_sum_dim
        self.head = nn.Linear(final_hidden_dim, self.target_size)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize all trainable parameters with Xavier uniform and zero biases."""
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

    @staticmethod
    def _to_device(tensor: Tensor, device: torch.device) -> Tensor:
        """Move a tensor to the target device if it is not already there."""
        if tensor.device == device:
            return tensor
        return tensor.to(device)

    def _embed_one_field(
        self,
        field_name: str,
        indices: Tensor,
        offsets: Tensor,
        weights: Tensor,
        *,
        batch_size: int,
        device: torch.device,
    ) -> Tensor:
        """Embed one field's sparse bag-of-tokens into a dense per-sample vector.

        :param indices: 1-D tensor of concatenated token indices across the batch
        :param offsets: 1-D tensor of start positions for each sample
        :param weights: 1-D tensor of per-token weights
        :return: Dense embedding tensor of shape [batch_size, embedding_dim]
        """
        indices = self._to_device(indices.long(), device)
        offsets = self._to_device(offsets.long(), device)
        weights = self._to_device(weights.float(), device)

        if offsets.dim() != 1 or offsets.size(0) != batch_size:
            raise ValueError(f"Field {field_name} offsets must have shape [batch_size], got {tuple(offsets.shape)}")
        
        if indices.dim() != 1:
            raise ValueError(f"Field {field_name} indices must be 1-D, got {tuple(indices.shape)}")
        
        if weights.dim() != 1:
            raise ValueError(f"Field {field_name} weights must be 1-D, got {tuple(weights.shape)}")
        
        if indices.size(0) != weights.size(0):
            raise ValueError(f"Field {field_name} indices/weights length mismatch: {indices.size(0)} vs {weights.size(0)}")

        return self.field_embeddings[field_name](indices, offsets, per_sample_weights=weights)

    def forward(self, feature_bags: Mapping[str, Mapping[str, Tensor]]) -> Tensor:
        """Forward pass.

        :param feature_bags: Per-field sparse bags with keys
            ``indices``, ``offsets``, ``weights``.
        :return: Multi-class logits with shape ``[batch_size, target_size]``
        """
        if not isinstance(feature_bags, Mapping) or not feature_bags:
            raise ValueError("feature_bags must be a non-empty mapping")

        # Infer batch size and device from the first field's offsets tensor
        first_field_name = self.field_names[0]
        first_offsets = feature_bags[first_field_name]["offsets"]
        batch_size = int(first_offsets.size(0))
        device = next(self.parameters()).device

        # Embed each field: sparse bag-of-tokens -> dense [batch_size, emb_dim] vector
        field_embeddings: dict[str, Tensor] = {}
        for field_name in self.field_names:
            if field_name not in feature_bags:
                raise ValueError(f"Missing feature bag for field {field_name}")

            bag = feature_bags[field_name]
            if not isinstance(bag, Mapping):
                raise ValueError(f"feature_bags[{field_name}] must be a mapping")

            # (batch_size, emb_dim)
            field_embeddings[field_name] = self._embed_one_field(
                field_name,
                bag["indices"],
                bag["offsets"],
                bag["weights"],
                batch_size=batch_size,
                device=device,
            )

        # Optionally reweight fields via learned attention, then concatenate
        if self.enable_attention:
            # [batch_size, field_count]
            field_scores = torch.cat(
                [
                    self.field_attention[field_name](field_embeddings[field_name])
                    for field_name in self.field_names
                ],
                dim=-1,
            )
            field_weights = torch.softmax(field_scores, dim=-1)
            weighted_embeddings = [
                field_embeddings[field_name] * field_weights[:, i].unsqueeze(-1)
                for i, field_name in enumerate(self.field_names)
            ]
            concat_embedding = torch.cat(weighted_embeddings, dim=-1)
        else:
            concat_embedding = torch.cat(
                [field_embeddings[field_name] for field_name in self.field_names],
                dim=-1,
            )

        if self.input_bn is not None:
            concat_embedding = self.input_bn(concat_embedding)

        hidden = self.mlp(concat_embedding)
        logits = self.head(hidden)
        return logits

    def forward_hidden(self, feature_bags: Mapping[str, Mapping[str, Tensor]]) -> Tensor:
        """Forward pass returning the MLP hidden state before the classification head.

        Useful for sampled softmax / NCE loss during training.

        :param feature_bags: Per-field sparse bags with keys
            ``indices``, ``offsets``, ``weights``.
        :return: Hidden state tensor of shape ``[batch_size, final_hidden_dim]``
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
