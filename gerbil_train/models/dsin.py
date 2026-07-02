"""DSIN (Deep Session Interest Network) for CTR prediction.

DSIN models user behavior as **sessions** rather than a flat sequence.

Architecture:
  1. Session Division: split behavior sequence into K sessions
  2. Bias Encoding: positional + session biases
  3. Bi-LSTM: extract interest from each session independently
  4. Multi-Head Self-Attention: model interactions between sessions
  5. Attention (DIN-style): target item attends to each session interest
"""

from __future__ import annotations

from typing import Mapping, Any

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from gerbil_train.config.model_config import DIENModelConfig, FieldEntry
from gerbil_train.utils.embedding import bag_to_padded, embed_one_field, to_device
from gerbil_train.models.layers import FullyConnectedLayer
from gerbil_train.models.base_model import BaseModel

__all__ = ["DSIN"]


class BiasEncoding(nn.Module):
    """Adds learnable session bias + position bias to session embeddings."""

    def __init__(self, num_sessions: int, session_len: int, emb_dim: int) -> None:
        super().__init__()
        self.session_bias = nn.Parameter(torch.zeros(1, num_sessions, 1, emb_dim))
        self.position_bias = nn.Parameter(torch.zeros(1, 1, session_len, emb_dim))

    def forward(self, x: Tensor) -> Tensor:
        # x: [B, num_sessions, session_len, emb_dim]
        return x + self.session_bias[:, :x.size(1)] + self.position_bias[:, :, :x.size(2)]


class DSIN(BaseModel):
    """Deep Session Interest Network for CTR prediction."""

    def __init__(self, model_cfg: DIENModelConfig) -> None:
        super().__init__()
        self._validate_fields(model_cfg)
        self.fields_cfg: Mapping[str, FieldEntry] = model_cfg.embedding_fields

        self.item_num = model_cfg.target_size
        self.behavior_fields = model_cfg.behavior_fields
        self.target_fields = model_cfg.target_fields
        reserved = set(self.behavior_fields) | set(self.target_fields)
        self.field_names = [n for n in self.fields_cfg if n not in reserved]

        self.emb_size = int(self.fields_cfg[self.behavior_fields[0]].emb_size)

        # Session config
        dsig_cfg: dict[str, Any] = model_cfg.interest_extractor
        self.num_sessions = int(dsig_cfg.get("num_sessions", 4))
        self.session_len = int(dsig_cfg.get("session_len", 10))
        self.lstm_hidden = int(dsig_cfg.get("lstm_hidden", 64))

        # Target embeddings
        self.target_embedding_bags = nn.ModuleDict()
        for f_name in self.target_fields:
            entry = self.fields_cfg[f_name]
            key = str(entry.field_index)
            if key not in self.target_embedding_bags:
                self.target_embedding_bags[key] = nn.EmbeddingBag(
                    num_embeddings=int(entry.dim),
                    embedding_dim=int(entry.emb_size),
                    mode="sum",
                )

        # Behavior embeddings (shared across all behavior fields)
        self.behavior_embeddings = nn.ModuleDict()
        for bf in self.behavior_fields:
            entry = self.fields_cfg[bf]
            self.behavior_embeddings[bf] = nn.Embedding(
                num_embeddings=int(entry.dim) + 1,
                embedding_dim=int(entry.emb_size),
                padding_idx=int(entry.dim),
            )

        # Plain field embeddings
        self.field_embedding_bags = nn.ModuleDict()
        for field_name in self.field_names:
            entry = self.fields_cfg[field_name]
            key = str(entry.field_index)
            if key not in self.field_embedding_bags:
                self.field_embedding_bags[key] = nn.EmbeddingBag(
                    num_embeddings=int(entry.dim),
                    embedding_dim=int(entry.emb_size),
                    mode="sum",
                )

        # 1. Bias Encoding
        self.bias_encoding = BiasEncoding(self.num_sessions, self.session_len, self.emb_size)

        # 2. Bi-LSTM per session
        self.session_lstm = nn.LSTM(
            input_size=self.emb_size,
            hidden_size=self.lstm_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        # 3. Multi-head self-attention across sessions
        attn_heads = int(dsig_cfg.get("attn_heads", 4))
        self.session_attention = nn.MultiheadAttention(
            embed_dim=self.lstm_hidden * 2,
            num_heads=attn_heads,
            batch_first=True,
        )

        # 4. DIN-style attention: target → each session interest
        attn_hidden = int(dsig_cfg.get("attn_hidden", 64))
        self.target_attention = nn.Sequential(
            nn.Linear(self.lstm_hidden * 2 + self.emb_size, attn_hidden),
            nn.ReLU(),
            nn.Linear(attn_hidden, 1),
        )

        # 5. MLP
        plain_dim = sum(int(self.fields_cfg[fn].emb_size) for fn in self.field_names)
        target_dim = sum(int(self.fields_cfg[tf].emb_size) for tf in self.target_fields)
        interest_dim = self.lstm_hidden * 2  # one interest vector after attention
        mlp_input_dim = plain_dim + target_dim + interest_dim

        mlp_cfg: dict[str, Any] = model_cfg.mlp
        hidden_dims = list(mlp_cfg.get("hidden_dims", [256, 128]))
        self.mlp = FullyConnectedLayer(
            input_dim=mlp_input_dim,
            hidden_dims=hidden_dims,
            bias=[True] * len(hidden_dims),
            batch_norm=bool(mlp_cfg.get("batch_norm", False)),
            activation=str(mlp_cfg.get("activation", "relu")),
            dropout=float(mlp_cfg.get("dropout", 0.0)),
        )
        final_dim = hidden_dims[-1] if hidden_dims else mlp_input_dim
        self.head = nn.Linear(final_dim, 1)
        self.reset_parameters()

    def _validate_fields(self, model_cfg: DIENModelConfig) -> None:
        if not model_cfg.embedding_fields:
            raise ValueError("embedding_fields must be a non-empty mapping")

    def reset_parameters(self) -> None:
        for emb in self.field_embedding_bags.values():
            nn.init.xavier_uniform_(emb.weight)
        for emb in self.target_embedding_bags.values():
            nn.init.xavier_uniform_(emb.weight)
        for emb in self.behavior_embeddings.values():
            nn.init.xavier_uniform_(emb.weight)
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, feature_bags: Mapping[str, Mapping[str, Tensor]]) -> Tensor:
        first_offsets = feature_bags[self.field_names[0]]["offsets"]
        batch_size = int(first_offsets.size(0))
        device = next(self.parameters()).device

        # 1. Plain field embeddings
        plain_embs: list[Tensor] = []
        for fn in self.field_names:
            entry = self.fields_cfg[fn]
            emb = embed_one_field(
                self.field_embedding_bags[str(entry.field_index)],
                feature_bags[fn]["indices"], feature_bags[fn]["offsets"],
                feature_bags[fn]["weights"], device=device,
            )
            plain_embs.append(emb)
        plain_concat = torch.cat(plain_embs, dim=-1) if plain_embs else torch.zeros(batch_size, 0, device=device)

        # 2. Target embeddings
        target_embs: list[Tensor] = []
        for tf in self.target_fields:
            entry = self.fields_cfg[tf]
            emb = embed_one_field(
                self.target_embedding_bags[str(entry.field_index)],
                feature_bags[tf]["indices"], feature_bags[tf]["offsets"],
                feature_bags[tf]["weights"], device=device,
            )
            target_embs.append(emb)
        target_concat = torch.cat(target_embs, dim=-1) if target_embs else torch.zeros(batch_size, 0, device=device)
        target_for_attn = torch.mean(torch.stack(target_embs, dim=0), dim=0) if target_embs else torch.zeros(batch_size, self.emb_size, device=device)

        # 3. Behavior sequence → sessions
        bf = self.behavior_fields[0]
        indices = to_device(feature_bags[bf]["indices"].long(), device)
        offsets = to_device(feature_bags[bf]["offsets"].long(), device)
        padded_ids, lengths, _ = bag_to_padded(indices, offsets)
        # padded_ids: [B, total_seq_len]

        # Split into sessions: [B, num_sessions, session_len]
        total_needed = self.num_sessions * self.session_len
        if padded_ids.size(1) < total_needed:
            padded_ids = F.pad(padded_ids, (0, total_needed - padded_ids.size(1)))
        else:
            padded_ids = padded_ids[:, :total_needed]
        session_ids = padded_ids.view(batch_size, self.num_sessions, self.session_len)

        # Embed sessions: [B, num_sessions, session_len, emb_dim]
        session_emb = self.behavior_embeddings[bf](session_ids)

        # Bias encoding
        session_emb = self.bias_encoding(session_emb)

        # 4. Bi-LSTM per session
        # Reshape: [B * num_sessions, session_len, emb_dim]
        B, S, L, D = session_emb.shape
        lstm_in = session_emb.view(B * S, L, D)
        lstm_out, _ = self.session_lstm(lstm_in)  # [B*S, L, 2*h]
        # Average pooling over session_len
        session_vec = lstm_out.mean(dim=1)          # [B*S, 2*h]
        session_vec = session_vec.view(B, S, -1)    # [B, S, 2*h]

        # 5. Self-attention across sessions
        attn_out, _ = self.session_attention(session_vec, session_vec, session_vec)  # [B, S, 2*h]
        session_vec = session_vec + attn_out                                         # residual

        # 6. DIN-style attention: target → session interests
        target_exp = target_for_attn.unsqueeze(1).expand(-1, S, -1)  # [B, S, emb_dim]
        attn_input = torch.cat([session_vec, target_exp], dim=-1)     # [B, S, 2*h + emb_dim]
        scores = self.target_attention(attn_input).squeeze(-1)        # [B, S]
        weights = torch.softmax(scores, dim=-1)                       # [B, S]
        interest = (weights.unsqueeze(-1) * session_vec).sum(dim=1)   # [B, 2*h]

        # 7. Concat → MLP
        combined = torch.cat([plain_concat, target_concat, interest], dim=-1)
        hidden = self.mlp(combined)
        logit = self.head(hidden).squeeze(-1)
        return torch.sigmoid(logit)
