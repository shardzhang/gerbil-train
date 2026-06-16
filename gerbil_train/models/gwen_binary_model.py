"""GwEN (Group-wise Embedding Network) for binary classification."""

from __future__ import annotations

from typing import Mapping

import torch
from torch import Tensor, nn

from gerbil_train.config import GwENModelConfig
from gerbil_train.utils.nn import build_mlp

__all__ = ["GwENBinary"]


class GwENBinary(nn.Module):
    """GwEN for binary classification."""

    def __init__(self, config: GwENModelConfig) -> None:
        super().__init__()
        fields_cfg = config.embedding_fields
        if not fields_cfg:
            raise ValueError("embedding_fields must be a non-empty mapping")

        self.field_names = list(fields_cfg.keys())
        self.field_embedding_dims: dict[str, int] = {}
        self.field_embeddings = nn.ModuleDict()
        for field_name in self.field_names:
            entry = fields_cfg[field_name]
            self.field_embedding_dims[field_name] = int(entry.emb_dim)
            self.field_embeddings[field_name] = nn.EmbeddingBag(
                num_embeddings=int(entry.vocab_size),
                embedding_dim=int(entry.emb_dim),
                mode="sum", include_last_offset=False,
            )

        self.enable_attention = bool(config.attention.get("enabled", False))
        if self.enable_attention:
            self.field_attention = nn.ModuleDict({
                fn: nn.Linear(self.field_embedding_dims[fn], 1, bias=False)
                for fn in self.field_names
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
        self.head = nn.Linear(final_hidden_dim, 1)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for emb in self.field_embeddings.values():
            nn.init.xavier_uniform_(emb.weight)
        if self.enable_attention:
            for lin in self.field_attention.values():
                nn.init.xavier_uniform_(lin.weight)
        for mod in self.mlp.modules():
            if isinstance(mod, nn.Linear):
                nn.init.xavier_uniform_(mod.weight)
                if mod.bias is not None:
                    nn.init.zeros_(mod.bias)
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    @staticmethod
    def _to_device(tensor: Tensor, device: torch.device) -> Tensor:
        return tensor if tensor.device == device else tensor.to(device)

    def _embed_one_field(self, field_name: str, indices: Tensor, offsets: Tensor,
                         weights: Tensor, *, batch_size: int, device: torch.device) -> Tensor:
        indices = self._to_device(indices.long(), device)
        offsets = self._to_device(offsets.long(), device)
        weights = self._to_device(weights.float(), device)
        return self.field_embeddings[field_name](indices, offsets, per_sample_weights=weights)

    def forward(self, feature_bags: Mapping[str, Mapping[str, Tensor]]) -> Tensor:
        """Forward pass returning binary probabilities.

        :return: Tensor of shape ``[batch_size]`` with values in ``[0, 1]``.
        """
        first_offsets = feature_bags[self.field_names[0]]["offsets"]
        batch_size = int(first_offsets.size(0))
        device = next(self.parameters()).device

        field_embeddings: dict[str, Tensor] = {}
        for fn in self.field_names:
            bag = feature_bags[fn]
            field_embeddings[fn] = self._embed_one_field(
                fn, bag["indices"], bag["offsets"], bag["weights"],
                batch_size=batch_size, device=device,
            )

        if self.enable_attention:
            scores = torch.cat([self.field_attention[fn](field_embeddings[fn]) for fn in self.field_names], dim=-1)
            w = torch.softmax(scores, dim=-1)
            emb = torch.cat([field_embeddings[fn] * w[:, i].unsqueeze(-1) for i, fn in enumerate(self.field_names)], dim=-1)
        else:
            emb = torch.cat([field_embeddings[fn] for fn in self.field_names], dim=-1)

        if self.input_bn is not None:
            emb = self.input_bn(emb)
        hidden = self.mlp(emb)
        return torch.sigmoid(self.head(hidden)).squeeze(-1)
