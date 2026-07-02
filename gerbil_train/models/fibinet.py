"""FiBiNet (Feature Importance and Bilinear Interaction Network) for CTR prediction.

FiBiNet = SENET (feature importance weighting)
         + Bilinear Interaction (v_i ⊙ W·v_j for all pairs)
         + MLP (Deep)

Key innovations:
  1. SENET: learns field-level importance via squeeze-excitation
  2. Bilinear Interaction: more expressive than dot product or FM
"""

from __future__ import annotations

from typing import Mapping, Any

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from gerbil_train.config.model_config import BaseModelConfig, FieldEntry
from gerbil_train.utils.embedding import embed_one_field
from gerbil_train.models.layers import FullyConnectedLayer
from gerbil_train.models.base_model import BaseModel

__all__ = ["FiBiNet"]


class SENETLayer(nn.Module):
    """Squeeze-and-Excitation Network for feature field importance weighting."""

    def __init__(self, num_fields: int, emb_dim: int, reduction: int = 3) -> None:
        super().__init__()
        squeezed = max(1, num_fields // reduction)
        self.excitation = nn.Sequential(
            nn.Linear(num_fields, squeezed, bias=False),
            nn.ReLU(),
            nn.Linear(squeezed, num_fields, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        # x: [B, n, d]
        Z = x.mean(dim=-1)           # [B, n]  squeeze
        A = self.excitation(Z)       # [B, n]  excitation
        return x * A.unsqueeze(-1)   # [B, n, d]  reweight


class BilinearInteraction(nn.Module):
    """Bilinear interaction between all field pairs.

    For each pair (i, j): p_ij = v_i ⊙ (W @ v_j)
    where W ∈ ℝ^{k×k} is a shared bilinear matrix.
    All pair outputs are element-wise summed: p = Σ_{i<j} v_i ⊙ (W @ v_j)
    """

    def __init__(self, emb_dim: int) -> None:
        super().__init__()
        self.W = nn.Parameter(torch.Tensor(emb_dim, emb_dim))
        nn.init.xavier_uniform_(self.W)

    def forward(self, x: Tensor) -> Tensor:
        B, n, k = x.shape
        # Shared bilinear: transform second embedding
        x_right = x @ self.W                # [B, n, k]
        all_pairs: list[Tensor] = []
        for i in range(n):
            for j in range(i + 1, n):
                pair = x[:, i, :] * x_right[:, j, :]  # [B, k]
                all_pairs.append(pair)
        return torch.stack(all_pairs, dim=0).sum(dim=0) if all_pairs else torch.zeros(B, k, device=x.device)


class FiBiNet(BaseModel):
    """Feature Importance and Bilinear Interaction Network for CTR prediction."""

    def __init__(self, model_cfg: BaseModelConfig) -> None:
        super().__init__()
        self._validate_fields(model_cfg)

        self.fields_cfg: Mapping[str, FieldEntry] = model_cfg.embedding_fields
        self.field_names = list(self.fields_cfg.keys())
        self.emb_size = int(next(iter(self.fields_cfg.values())).emb_size)

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
                    embedding_dim=int(self.emb_size),
                    mode="sum",
                )

        # Number of fields that participate in interactions
        n_emb = sum(1 for e in self.fields_cfg.values() if not (e.field_type == 0 and e.concat_type == "direct"))

        # SENET Layer (feature importance)
        fibi_cfg: dict[str, Any] = model_cfg.field_attention
        senet_reduction = int(fibi_cfg.get("senet_reduction", 3))
        self.senet = SENETLayer(n_emb, self.emb_size, senet_reduction)

        # Bilinear Interaction (shared W matrix)
        self.bilinear_orig = BilinearInteraction(self.emb_size)
        self.bilinear_senet = BilinearInteraction(self.emb_size)

        # MLP on concat(original_interaction, senet_interaction)
        interaction_dim = self.emb_size * 2  # orig + senet bilinear
        mlp_cfg: dict[str, Any] = model_cfg.mlp
        hidden_dims = list(mlp_cfg.get("hidden_dims", [256, 128]))
        self.mlp = FullyConnectedLayer(
            input_dim=interaction_dim,
            hidden_dims=hidden_dims,
            bias=[True] * len(hidden_dims),
            batch_norm=bool(mlp_cfg.get("batch_norm", False)),
            activation=str(mlp_cfg.get("activation", "relu")),
            dropout=float(mlp_cfg.get("dropout", 0.0)),
        )
        final_dim = hidden_dims[-1] if hidden_dims else interaction_dim
        self.head = nn.Linear(final_dim, 1)
        self.bias = nn.Parameter(torch.zeros(1))
        self.reset_parameters()

    def _validate_fields(self, model_cfg: BaseModelConfig) -> None:
        if not model_cfg.embedding_fields:
            raise ValueError("embedding_fields must be a non-empty mapping")
        emb_sizes = {int(e.emb_size) for e in model_cfg.embedding_fields.values()
                     if not (e.field_type == 0 and e.concat_type == "direct")}
        if len(emb_sizes) > 1:
            raise ValueError(f"FiBiNet requires all field embeddings to have the same size, got {emb_sizes}")

    def reset_parameters(self) -> None:
        nn.init.zeros_(self.bias)
        for emb in self.linear_embeddings.values():
            nn.init.xavier_uniform_(emb.weight)
        for emb in self.feature_embeddings.values():
            nn.init.xavier_uniform_(emb.weight)
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, feature_bags: Mapping[str, Mapping[str, Tensor]]) -> Tensor:
        first_offsets = feature_bags[self.field_names[0]]["offsets"]
        batch_size = int(first_offsets.size(0))
        device = next(self.parameters()).device

        # 1. Linear term
        linear_sum = self.bias.expand(batch_size).to(device)
        emb_list: list[Tensor] = []
        for field_name, entry in self.fields_cfg.items():
            if entry.field_type == 0 and entry.concat_type == "direct":
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

        # 2. Stack embeddings
        x = torch.stack(emb_list, dim=1)         # [B, n, k]

        # 3. SENET → reweighted embeddings
        x_senet = self.senet(x)                  # [B, n, k]

        # 4. Bilinear interactions (original + senet)
        bi_orig = self.bilinear_orig(x)          # [B, k]
        bi_senet = self.bilinear_senet(x_senet)  # [B, k]

        # 5. MLP on interactions
        interaction = torch.cat([bi_orig, bi_senet], dim=-1)  # [B, 2k]
        hidden = self.mlp(interaction)
        deep_logit = self.head(hidden).squeeze(-1)

        return torch.sigmoid(linear_sum + deep_logit)
