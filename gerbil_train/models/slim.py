"""SLIM (Sparse Linear Method) for CTR prediction.

SLIM = Linear (1st-order) with sparsity-inducing L1 regularization.

$$ \hat{y} = \text{sigmoid}(w_0 + \sum_i w_i x_i) $$
"""

from __future__ import annotations

from typing import Mapping

import torch
from torch import Tensor, nn

from gerbil_train.config.model_config import BaseModelConfig, FieldEntry
from gerbil_train.utils.embedding import embed_one_field
from gerbil_train.models.base_model import BaseModel

__all__ = ["SLIM"]


class SLIM(BaseModel):
    """SLIM model for CTR prediction."""

    def __init__(self, model_cfg: BaseModelConfig) -> None:
        super().__init__()

        self._validate_fields(model_cfg)

        self.fields_cfg: Mapping[str, FieldEntry] = model_cfg.embedding_fields
        self.field_names = list(self.fields_cfg.keys())

        # Linear embeddings: vocab → 1 (one weight per feature value)
        self.linear_embedding_bags = nn.ModuleDict()
        for field_name, entry in self.fields_cfg.items():
            key = str(entry.field_index)
            if key not in self.linear_embedding_bags:
                self.linear_embedding_bags[key] = nn.EmbeddingBag(
                    num_embeddings=int(entry.dim),
                    embedding_dim=1,
                    mode="sum",
                )

        self.bias = nn.Parameter(torch.zeros(1))
        self.reset_parameters()

    def _validate_fields(self, model_cfg: BaseModelConfig) -> None:
        if not model_cfg.embedding_fields:
            raise ValueError("embedding_fields must be a non-empty mapping")

    def reset_parameters(self) -> None:
        nn.init.zeros_(self.bias)
        for emb in self.linear_embedding_bags.values():
            nn.init.xavier_uniform_(emb.weight)

    def forward(self, feature_bags: Mapping[str, Mapping[str, Tensor]]) -> Tensor:
        first_offsets = feature_bags[self.field_names[0]]["offsets"]
        batch_size = int(first_offsets.size(0))
        device = next(self.parameters()).device

        # Linear term: w_0 + Σ w_i · x_i
        linear_sum = self.bias.expand(batch_size).to(device)
        for field_name, entry in self.fields_cfg.items():
            linear_emb = embed_one_field(
                self.linear_embedding_bags[str(entry.field_index)],
                feature_bags[field_name]["indices"],
                feature_bags[field_name]["offsets"],
                feature_bags[field_name]["weights"],
                device=device,
            )
            linear_sum = linear_sum + linear_emb.squeeze(-1)

        logits = linear_sum
        return torch.sigmoid(logits)
