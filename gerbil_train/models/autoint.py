"""AutoInt (Automatic Feature Interaction) via multi-head self-attention.

AutoInt uses stacked Transformer encoder layers to learn feature
interactions. Each feature field is treated as a "token", and multi-head
self-attention learns which fields interact and how they interact.

Architecture:
  Embedding → [Interacting Layer × N] (MHA + FFN) → Concat → MLP → Output
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

__all__ = ["AutoInt"]


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention over feature field embeddings."""

    def __init__(self, emb_dim: int, num_heads: int, attn_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        assert attn_dim % num_heads == 0, "attn_dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = attn_dim // num_heads

        self.q_proj = nn.Linear(emb_dim, attn_dim, bias=False)
        self.k_proj = nn.Linear(emb_dim, attn_dim, bias=False)
        self.v_proj = nn.Linear(emb_dim, attn_dim, bias=False)
        self.res_proj = nn.Linear(attn_dim, emb_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        B, n, d = x.shape
        Q = self.q_proj(x).view(B, n, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(B, n, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(B, n, self.num_heads, self.head_dim).transpose(1, 2)

        attn = (Q @ K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = (attn @ V).transpose(1, 2).reshape(B, n, -1)
        return self.res_proj(out)


class InteractingLayer(nn.Module):
    """One AutoInt interacting layer: MHA → Residual → FFN → Residual."""

    def __init__(self, emb_dim: int, num_heads: int, attn_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.attention = MultiHeadSelfAttention(emb_dim, num_heads, attn_dim, dropout)
        self.ffn = nn.Sequential(
            nn.Linear(emb_dim, emb_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(emb_dim * 2, emb_dim),
        )
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(emb_dim)
        self.norm2 = nn.LayerNorm(emb_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.dropout(self.attention(self.norm1(x)))
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x


class AutoInt(BaseModel):
    """Automatic Feature Interaction Network via Multi-Head Self-Attention."""

    def __init__(self, model_cfg: BaseModelConfig) -> None:
        super().__init__()
        self._validate_fields(model_cfg)

        self.fields_cfg: Mapping[str, FieldEntry] = model_cfg.embedding_fields
        self.field_names = list(self.fields_cfg.keys())
        self.emb_size = int(next(iter(model_cfg.embedding_fields.values())).emb_size)

        # Field embeddings (shared for linear + attention)
        self.field_embeddings = nn.ModuleDict()
        # Linear embeddings: vocab → 1
        self.linear_embeddings = nn.ModuleDict()

        for field_name, entry in self.fields_cfg.items():
            if entry.field_type == 0 and entry.concat_type == "direct":
                continue
            key = str(entry.field_index)

            if key not in self.field_embeddings:
                self.field_embeddings[key] = nn.EmbeddingBag(
                    num_embeddings=int(entry.dim),
                    embedding_dim=int(self.emb_size),
                    mode="sum",
                )
            if key not in self.linear_embeddings:
                self.linear_embeddings[key] = nn.EmbeddingBag(
                    num_embeddings=int(entry.dim),
                    embedding_dim=1,
                    mode="sum",
                )

        # AutoInt config
        acfg: dict[str, Any] = model_cfg.auto_attention
        num_layers = int(acfg.get("num_layers", 3))
        num_heads = int(acfg.get("num_heads", 2))
        attn_dim = int(acfg.get("attn_dim", 32))
        attn_dropout = float(acfg.get("dropout", 0.0))

        # Stacked interacting layers (Transformer encoder)
        self.layers = nn.ModuleList([
            InteractingLayer(self.emb_size, num_heads, attn_dim, attn_dropout)
            for _ in range(num_layers)
        ])

        # MLP on top of concatenated field outputs
        n_emb = sum(1 for e in self.fields_cfg.values() if not (e.field_type == 0 and e.concat_type == "direct"))
        mlp_cfg: dict[str, Any] = model_cfg.mlp
        hidden_dims = list(mlp_cfg.get("hidden_dims", [128]))
        self.mlp = FullyConnectedLayer(
            input_dim=n_emb * self.emb_size,
            hidden_dims=hidden_dims,
            bias=[True] * len(hidden_dims),
            batch_norm=bool(mlp_cfg.get("batch_norm", False)),
            activation=str(mlp_cfg.get("activation", "relu")),
            dropout=float(mlp_cfg.get("dropout", 0.0)),
        )
        final_dim = hidden_dims[-1] if hidden_dims else n_emb * self.emb_size
        self.head = nn.Linear(final_dim, 1)
        self.bias = nn.Parameter(torch.zeros(1))
        self.reset_parameters()

    def _validate_fields(self, model_cfg: BaseModelConfig) -> None:
        if not model_cfg.embedding_fields:
            raise ValueError("embedding_fields must be a non-empty mapping")
        emb_sizes = {int(e.emb_size) for e in model_cfg.embedding_fields.values()
                     if not (e.field_type == 0 and e.concat_type == "direct")}
        if len(emb_sizes) > 1:
            raise ValueError(f"AutoInt requires all field embeddings to have the same size, got {emb_sizes}")

    def reset_parameters(self) -> None:
        nn.init.zeros_(self.bias)
        for emb in self.linear_embeddings.values():
            nn.init.xavier_uniform_(emb.weight)
        for emb in self.field_embeddings.values():
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

            field_emb = embed_one_field(
                self.field_embeddings[key],
                feature_bags[field_name]["indices"],
                feature_bags[field_name]["offsets"],
                feature_bags[field_name]["weights"],
                device=device,
            )
            emb_list.append(field_emb)

        # 2. Stacked Interacting Layers (Transformer encoder)
        x = torch.stack(emb_list, dim=1)   # [B, n, d]
        for layer in self.layers:
            x = layer(x)                   # [B, n, d]

        # 3. Concat all field outputs → MLP → head
        concat = x.reshape(batch_size, -1)   # [B, n*d]
        deep_logit = self.head(self.mlp(concat)).squeeze(-1)

        return torch.sigmoid(linear_sum + deep_logit)
