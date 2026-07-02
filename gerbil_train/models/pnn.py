"""PNN (Product-based Neural Network) for CTR prediction.

PNN = Linear (1st-order) + Product Layer (pair-wise inner products) + MLP (Deep).

The Product Layer captures pair-wise feature interactions explicitly:
    l_z = [v_1, ..., v_n]          (concatenated embeddings)
    l_p = [⟨v_1,v₂⟩, ..., ⟨v_{n-1},v_n⟩]  (inner products of all pairs)
    hidden = MLP(concat(l_z, l_p))
"""

from __future__ import annotations

from typing import Mapping, Any

import torch
from torch import Tensor, nn

from gerbil_train.config.model_config import BaseModelConfig, FieldEntry
from gerbil_train.utils.embedding import embed_one_field
from gerbil_train.models.layers import FullyConnectedLayer
from gerbil_train.models.base_model import BaseModel

__all__ = ["PNN"]


class PNN(BaseModel):
    """Product-based Neural Network for CTR prediction."""

    def __init__(self, model_cfg: BaseModelConfig) -> None:
        super().__init__()
        self._validate_fields(model_cfg)

        self.fields_cfg: Mapping[str, FieldEntry] = model_cfg.embedding_fields
        self.field_names = list(self.fields_cfg.keys())
        self.num_fields = len(self.field_names)
        self.emb_size = int(next(iter(self.fields_cfg.values())).emb_size)

        # Linear embeddings: vocab → 1
        self.linear_embeddings = nn.ModuleDict()
        # Feature embeddings: vocab → k
        self.product_embeddings = nn.ModuleDict()

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
            if key not in self.product_embeddings:
                self.product_embeddings[key] = nn.EmbeddingBag(
                    num_embeddings=int(entry.dim),
                    embedding_dim=int(self.emb_size),
                    mode="sum",
                )

        # MLP on concatenated linear embeddings + all pair-wise inner products
        # Only non-direct fields participate in embeddings
        n_emb = sum(1 for e in self.fields_cfg.values() if not (e.field_type == 0 and e.concat_type == "direct"))
        num_pairs = n_emb * (n_emb - 1) // 2
        product_dim = n_emb * self.emb_size + num_pairs

        mlp_cfg: dict[str, Any] = model_cfg.mlp
        hidden_dims = list(mlp_cfg.get("hidden_dims", [128, 64]))
        self.mlp = FullyConnectedLayer(
            input_dim=product_dim,
            hidden_dims=hidden_dims,
            bias=[True] * len(hidden_dims),
            batch_norm=bool(mlp_cfg.get("batch_norm", False)),
            activation=str(mlp_cfg.get("activation", "relu")),
            dropout=float(mlp_cfg.get("dropout", 0.0)),
        )
        final_dim = hidden_dims[-1] if hidden_dims else product_dim
        self.head = nn.Linear(final_dim, 1)
        self.bias = nn.Parameter(torch.zeros(1))
        self.reset_parameters()

    def _validate_fields(self, model_cfg: BaseModelConfig) -> None:
        if not model_cfg.embedding_fields:
            raise ValueError("embedding_fields must be a non-empty mapping")
        emb_sizes = {int(e.emb_size) for e in model_cfg.embedding_fields.values()
                     if not (e.field_type == 0 and e.concat_type == "direct")}
        if len(emb_sizes) > 1:
            raise ValueError(f"PNN requires all field embeddings to have the same size, got {emb_sizes}")

    def reset_parameters(self) -> None:
        nn.init.zeros_(self.bias)
        for emb in self.linear_embeddings.values():
            nn.init.xavier_uniform_(emb.weight)
        for emb in self.product_embeddings.values():
            nn.init.xavier_uniform_(emb.weight)
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, feature_bags: Mapping[str, Mapping[str, Tensor]]) -> Tensor:
        first_offsets = feature_bags[self.field_names[0]]["offsets"]
        batch_size = int(first_offsets.size(0))
        device = next(self.parameters()).device

        # 1. Linear term: w_0 + Σ w_i · x_i
        linear_sum = self.bias.expand(batch_size).to(device)
        emb_list: list[Tensor] = []
        for field_name, entry in self.fields_cfg.items():
            if entry.field_type == 0 and entry.concat_type == "direct":
                continue
            linear_emb = embed_one_field(
                self.linear_embeddings[str(entry.field_index)],
                feature_bags[field_name]["indices"],
                feature_bags[field_name]["offsets"],
                feature_bags[field_name]["weights"],
                device=device,
            )
            linear_sum = linear_sum + linear_emb.squeeze(-1)

            feature_emb = embed_one_field(
                self.product_embeddings[str(entry.field_index)],
                feature_bags[field_name]["indices"],
                feature_bags[field_name]["offsets"],
                feature_bags[field_name]["weights"],
                device=device,
            )
            emb_list.append(feature_emb)

        # 2. Product layer: concat(z) + concat(inner products)
        # l_z = [v_1, ..., v_n]  (concatenation, preserves per-field info)
        l_z = torch.cat(emb_list, dim=-1)                       # [B, n·k]

        # l_p = [⟨v_i, v_j⟩] for all i < j (pair-wise inner products)
        n = len(emb_list)
        pairs: list[Tensor] = []
        for i in range(n):
            for j in range(i + 1, n):
                ip = (emb_list[i] * emb_list[j]).sum(dim=-1, keepdim=True)  # [B, 1]
                pairs.append(ip)
        l_p = torch.cat(pairs, dim=-1) if pairs else torch.zeros(batch_size, 0, device=device)  # [B, n_pairs]

        hidden = self.mlp(torch.cat([l_z, l_p], dim=-1))        # [B, h]
        deep_logit = self.head(hidden).squeeze(-1)               # [B]

        return torch.sigmoid(linear_sum + deep_logit)
