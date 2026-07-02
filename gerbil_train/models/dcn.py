"""DCN (Deep & Cross Network) for CTR prediction.

DCN = Cross Network (explicit bounded-degree feature interactions)
      + Deep Network (MLP)
      + Linear (1st-order)

     ┌─────────────────────────────────┐
     │         Output (sigmoid)        │
     └────────────────┬────────────────┘
                      │
             ┌────────┴────────┐
             │   Combination   │
             │    Linear(·)    │
             └────────┬────────┘
                      │
         ┌────────────┴────────────┐
         │      concat(c, d)       │
         └────────────┬────────────┘
         │                         │
   ┌─────┴─────┐           ┌──────┴──────┐
   │   Cross   │           │    Deep     │
   │  Network  │           │    MLP      │
   └─────┬─────┘           └──────┬──────┘
         │                         │
         └──────────┬──────────────┘
                    │
          ┌─────────┴─────────┐
          │   Field Embedding │
          │    concat + BN    │
          └─────────┬─────────┘
                    │
              feature_bags
"""

from __future__ import annotations

from typing import Mapping, Any

import torch
from torch import Tensor, nn

from gerbil_train.config.model_config import BaseModelConfig, FieldEntry
from gerbil_train.utils.embedding import embed_one_field
from gerbil_train.models.layers import FullyConnectedLayer
from gerbil_train.models.base_model import BaseModel

__all__ = ["DCN"]


class CrossNetwork(nn.Module):
    """Cross Network from DCN (Deep & Cross Network).

    Each layer computes: x_{l+1} = x_0 ⊙ (W_l · x_l + b_l) + x_l
    where ⊙ is element-wise multiplication.
    """

    def __init__(self, input_dim: int, num_layers: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.layers = nn.ModuleList([
            nn.Linear(input_dim, input_dim, bias=True) for _ in range(num_layers)
        ])

    def forward(self, x_0: Tensor) -> Tensor:
        x_l = x_0
        for layer in self.layers:
            x_l = x_0 * layer(x_l) + x_l
        return x_l


class DCN(BaseModel):
    """Deep & Cross Network for CTR prediction."""

    def __init__(self, model_cfg: BaseModelConfig) -> None:
        super().__init__()
        self._validate_fields(model_cfg)

        self.fields_cfg: Mapping[str, FieldEntry] = model_cfg.embedding_fields
        self.field_names = list(self.fields_cfg.keys())

        # Compute embedding dim, excluding direct fields
        emb_dims = {}
        for field_name, entry in self.fields_cfg.items():
            if entry.field_type == 0 and entry.concat_type == "direct":
                emb_dims[field_name] = int(entry.dim)
                continue
            emb_dims[field_name] = int(entry.emb_size)
        self.embedding_sum_dim = sum(emb_dims.values())

        # Linear embeddings: vocab → 1
        self.linear_embeddings = nn.ModuleDict()
        # Feature embeddings: vocab → k
        self.feature_embeddings = nn.ModuleDict()

        for field_name, entry in self.fields_cfg.items():
            if entry.field_type == 0 and entry.concat_type == "direct":
                continue
            key = str(entry.field_index)
            if key not in self.linear_embeddings:
                self.linear_embeddings[key] = nn.EmbeddingBag(
                    num_embeddings=int(entry.dim),
                    embedding_dim=1,
                    mode="sum",
                )
            if key not in self.feature_embeddings:
                self.feature_embeddings[key] = nn.EmbeddingBag(
                    num_embeddings=int(entry.dim),
                    embedding_dim=int(entry.emb_size),
                    mode="sum",
                )

        # Cross Network
        cross_cfg: dict[str, Any] = model_cfg.field_attention
        num_cross_layers = int(cross_cfg.get("num_cross_layers", 3))
        self.cross_network = CrossNetwork(self.embedding_sum_dim, num_cross_layers)

        # Deep Network
        mlp_cfg: dict[str, Any] = model_cfg.mlp
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

        # Combination layer: concat(cross_output, deep_output) → 1
        self.combine = nn.Linear(self.embedding_sum_dim + deep_output_dim, 1)
        self.bias = nn.Parameter(torch.zeros(1))
        self.reset_parameters()

    def _validate_fields(self, model_cfg: BaseModelConfig) -> None:
        if not model_cfg.embedding_fields:
            raise ValueError("embedding_fields must be a non-empty mapping")

    def reset_parameters(self) -> None:
        nn.init.zeros_(self.bias)
        for emb in self.linear_embeddings.values():
            nn.init.xavier_uniform_(emb.weight)
        for emb in self.feature_embeddings.values():
            nn.init.xavier_uniform_(emb.weight)

    def forward(self, feature_bags: Mapping[str, Mapping[str, Tensor]]) -> Tensor:
        first_offsets = feature_bags[self.field_names[0]]["offsets"]
        batch_size = int(first_offsets.size(0))
        device = next(self.parameters()).device

        # 1. Linear term
        linear_sum = self.bias.expand(batch_size).to(device)
        emb_list: list[Tensor] = []
        for field_name, entry in self.fields_cfg.items():
            if entry.field_type == 0 and entry.concat_type == "direct":
                emb_list.append(feature_bags[field_name]["weights"].view(-1, int(entry.dim)))
                continue
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
                self.feature_embeddings[key],
                feature_bags[field_name]["indices"],
                feature_bags[field_name]["offsets"],
                feature_bags[field_name]["weights"],
                device=device,
            )
            emb_list.append(feature_emb)

        x_0 = torch.cat(emb_list, dim=-1)               # [B, d]

        # 2. Cross Network
        cross_out = self.cross_network(x_0)              # [B, d]

        # 3. Deep Network
        deep_out = self.deep_network(x_0)                # [B, h]

        # 4. Combine → output
        logits = self.combine(torch.cat([cross_out, deep_out], dim=-1)).squeeze(-1)  # [B]
        return torch.sigmoid(linear_sum + logits)
