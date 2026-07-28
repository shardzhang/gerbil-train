"""NCF (Neural Collaborative Filtering) for CTR prediction.

NCF replaces the inner product in Matrix Factorization with a neural
architecture consisting of GMF (Generalized Matrix Factorization) and
MLP (Multi-Layer Perceptron) paths, fused via NeuMF at the vector level.

Architecture (NeuMF):
  GMF path:  p_u^G ⊙ q_i^G  →  φ_GMF  (vector)
  MLP path:  [p_u^M; q_i^M] → MLP → φ_MLP  (vector)
  Fusion:    [φ_GMF; φ_MLP; plain] → h^T · φ → σ

GMF and MLP use SEPARATE user/item embeddings (per paper Section 3.4).

Reference: https://arxiv.org/abs/1708.05031 (WWW 2017)
"""

from __future__ import annotations

from typing import Mapping, Any

import torch
from torch import Tensor, nn

from gerbil_train.config.model_config import BaseModelConfig, FieldEntry
from gerbil_train.utils.embedding import embed_one_field
from gerbil_train.models.layers import FullyConnectedLayer
from gerbil_train.models.base_model import BaseModel

__all__ = ["NCF"]


class NCF(BaseModel):
    """Neural Collaborative Filtering with vector-level NeuMF fusion."""

    def __init__(self, model_cfg: BaseModelConfig) -> None:
        super().__init__()

        self._validate_fields(model_cfg)

        self.embedding_fields: Mapping[str, FieldEntry] = model_cfg.embedding_fields
        self.field_names = list(self.embedding_fields.keys())

        # Designate user/item fields from mlp config
        ncf_cfg = dict(model_cfg.mlp)
        self.user_field: str = str(ncf_cfg.pop("user_field", self.field_names[0]))
        self.item_field: str = str(ncf_cfg.pop("item_field", self.field_names[1]))
        self.plain_field_names = [n for n in self.field_names if n not in (self.user_field, self.item_field)]
        user_entry = self.embedding_fields[self.user_field]
        item_entry = self.embedding_fields[self.item_field]
        self.emb_size = int(user_entry.emb_size)

        # --- Separate embedding bags ---

        # Plain (non-user, non-item) field embeddings
        self.plain_embedding_bags = nn.ModuleDict()
        for fn in self.plain_field_names:
            entry = self.embedding_fields[fn]
            if entry.field_type == 0 and entry.concat_type == "direct":
                continue
            key = str(entry.field_index)
            if key not in self.plain_embedding_bags:
                self.plain_embedding_bags[key] = nn.EmbeddingBag(
                    num_embeddings=int(entry.dim),
                    embedding_dim=int(entry.emb_size),
                    mode="sum",
                )

        # GMF path: user and item embeddings (separate from MLP)
        self.gmf_user_embedding = nn.EmbeddingBag(
            num_embeddings=int(user_entry.dim),
            embedding_dim=int(user_entry.emb_size),
            mode="sum",
        )
        self.gmf_item_embedding = nn.EmbeddingBag(
            num_embeddings=int(item_entry.dim),
            embedding_dim=int(item_entry.emb_size),
            mode="sum",
        )

        # MLP path: separate user and item embeddings
        self.mlp_user_embedding = nn.EmbeddingBag(
            num_embeddings=int(user_entry.dim),
            embedding_dim=int(user_entry.emb_size),
            mode="sum",
        )
        self.mlp_item_embedding = nn.EmbeddingBag(
            num_embeddings=int(item_entry.dim),
            embedding_dim=int(item_entry.emb_size),
            mode="sum",
        )

        # --- GMF path ---
        # φ_GMF = p_u^G ⊙ q_i^G  (no linear head, direct vector output)

        # --- MLP path ---
        # z_1 = [p_u^M; q_i^M] → MLP → φ_MLP (last hidden layer)
        mlp_hidden = list(ncf_cfg["hidden_dims"])
        self.mlp_layers = FullyConnectedLayer(
            input_dim=self.emb_size * 2,
            hidden_dims=mlp_hidden,
            bias=[True] * len(mlp_hidden),
            batch_norm=bool(ncf_cfg["batch_norm"]),
            activation=str(ncf_cfg["activation"]),
            dropout=float(ncf_cfg["dropout"]),
        )

        # --- Plain features ---
        plain_dim = 0
        for fn in self.plain_field_names:
            entry = self.embedding_fields[fn]
            if entry.field_type == 0 and entry.concat_type == "direct":
                plain_dim += int(entry.dim)
            else:
                plain_dim += int(entry.emb_size)

        # --- NeuMF fusion ---
        # φ = [φ_GMF; φ_MLP]  → h^T · φ → σ
        fusion_dim = self.emb_size + mlp_hidden[-1] + plain_dim
        self.fusion_head = nn.Linear(fusion_dim, 1, bias=False)

        self.reset_parameters()

    def _validate_fields(self, model_cfg: BaseModelConfig) -> None:
        if not model_cfg.embedding_fields:
            raise ValueError("embedding_fields must be a non-empty mapping")

    def reset_parameters(self) -> None:
        for emb in self.plain_embedding_bags.values():
            nn.init.xavier_uniform_(emb.weight)
        nn.init.xavier_uniform_(self.gmf_user_embedding.weight)
        nn.init.xavier_uniform_(self.gmf_item_embedding.weight)
        nn.init.xavier_uniform_(self.mlp_user_embedding.weight)
        nn.init.xavier_uniform_(self.mlp_item_embedding.weight)
        nn.init.xavier_uniform_(self.fusion_head.weight)

    def forward(self, feature_bags: Mapping[str, Mapping[str, Tensor]]) -> Tensor:
        first_offsets = feature_bags[next(iter(self.field_names))]["offsets"]
        batch_size = int(first_offsets.size(0))
        device = next(self.parameters()).device

        # --- Embed all plain fields ---
        plain_embs: list[Tensor] = []
        for fn in self.plain_field_names:
            entry = self.embedding_fields[fn]
            if entry.field_type == 0 and entry.concat_type == "direct":
                plain_embs.append(feature_bags[fn]["weights"].view(-1, int(entry.dim)))
            else:
                plain_embs.append(embed_one_field(
                    self.plain_embedding_bags[str(entry.field_index)],
                    feature_bags[fn]["indices"],
                    feature_bags[fn]["offsets"],
                    feature_bags[fn]["weights"],
                    device=device,
                ))
        plain_concat = torch.cat(plain_embs, dim=-1) if plain_embs else torch.zeros(batch_size, 0, device=device)

        # --- GMF path: φ_GMF = p_u^G ⊙ q_i^G (vector) ---
        user_emb_g = embed_one_field(
            self.gmf_user_embedding,
            feature_bags[self.user_field]["indices"],
            feature_bags[self.user_field]["offsets"],
            feature_bags[self.user_field]["weights"],
            device=device,
        )
        item_emb_g = embed_one_field(
            self.gmf_item_embedding,
            feature_bags[self.item_field]["indices"],
            feature_bags[self.item_field]["offsets"],
            feature_bags[self.item_field]["weights"],
            device=device,
        )
        phi_gmf = user_emb_g * item_emb_g                                   # [B, d]

        # --- MLP path: φ_MLP = MLP([p_u^M; q_i^M]) (vector) ---
        user_emb_m = embed_one_field(
            self.mlp_user_embedding,
            feature_bags[self.user_field]["indices"],
            feature_bags[self.user_field]["offsets"],
            feature_bags[self.user_field]["weights"],
            device=device,
        )
        item_emb_m = embed_one_field(
            self.mlp_item_embedding,
            feature_bags[self.item_field]["indices"],
            feature_bags[self.item_field]["offsets"],
            feature_bags[self.item_field]["weights"],
            device=device,
        )
        mlp_input = torch.cat([user_emb_m, item_emb_m], dim=-1)            # [B, 2d]
        phi_mlp = self.mlp_layers(mlp_input)                                # [B, h]

        # --- NeuMF fusion: h^T · [φ_GMF; φ_MLP; plain] (no bias) ---
        fusion_input = torch.cat([phi_gmf, phi_mlp, plain_concat], dim=-1)
        logit = self.fusion_head(fusion_input).squeeze(-1)
        return torch.sigmoid(logit)
