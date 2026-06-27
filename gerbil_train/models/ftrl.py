"""FTRL (Follow The Regularized Leader) linear model for CTR prediction.

FTRL is a linear model where each categorical field is embedded via EmbeddingBag(vocab → 1).
The output is a simple weighted sum: Σ(w_i · x_i) + bias → sigmoid.
The key difference is the FTRL-Proximal optimizer used during training.
"""

from __future__ import annotations

from typing import Mapping

import torch
from torch import Tensor, nn

from gerbil_train.config.model_config import BaseModelConfig, FieldEntry
from gerbil_train.utils.embedding import embed_one_field
from gerbil_train.models.base_model import BaseModel

__all__ = ["FTRLModel"]


class FTRLModel(BaseModel):
    """FTRL linear model with per-field EmbeddingBag(vocab → 1).

    Forward: sum of all field embeddings + bias → sigmoid.

    :param model_cfg: Model configuration with embedding_fields
    """

    def __init__(self, model_cfg: BaseModelConfig) -> None:
        super().__init__()
        self._validate_fields(model_cfg)

        self.fields_cfg: Mapping[str, FieldEntry] = model_cfg.embedding_fields
        self.field_names = list(self.fields_cfg.keys())

        self.linear_embeddings = nn.ModuleDict()
        for field_name, entry in self.fields_cfg.items():
            key = str(entry.field_index)
            if key not in self.linear_embeddings:
                bag = nn.EmbeddingBag(
                    num_embeddings=int(entry.dim),
                    embedding_dim=1,
                    mode="sum",
                )
                bag.field_name = field_name
                self.linear_embeddings[key] = bag

        self.bias = nn.Parameter(torch.zeros(1))
        self.reset_parameters()

    def _validate_fields(self, model_cfg: BaseModelConfig) -> None:
        if not model_cfg.embedding_fields:
            raise ValueError("embedding_fields must be a non-empty mapping")

    def reset_parameters(self) -> None:
        nn.init.zeros_(self.bias)
        for emb in self.linear_embeddings.values():
            nn.init.zeros_(emb.weight)

    def forward(self, feature_bags: Mapping[str, Mapping[str, Tensor]]) -> Tensor:
        first_offsets = feature_bags[self.field_names[0]]["offsets"]
        batch_size = int(first_offsets.size(0))
        device = next(self.parameters()).device

        logits = self.bias.expand(batch_size).to(device)
        for field_name, entry in self.fields_cfg.items():
            linear_emb = embed_one_field(
                self.linear_embeddings[str(entry.field_index)],
                feature_bags[field_name]["indices"],
                feature_bags[field_name]["offsets"],
                feature_bags[field_name]["weights"],
                device=device,
            )
            logits = logits + linear_emb.squeeze(-1)

        return torch.sigmoid(logits)
