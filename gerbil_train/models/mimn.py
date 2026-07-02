"""MIMN (Multi-channel Interest with Moment Network) for CTR prediction.

MIMN uses a multi-slot memory network to capture multiple aspects of user
interests, unlike DIN/DIEN which produce a single interest vector.

Architecture:
  1. Behavior sequence → Bi-LSTM → sequence of interest states
  2. Memory Write: each state writes to K memory slots via attention
  3. Target Read: target item reads from memory via attention
  4. Interest = weighted sum of memory slots
  5. Concat(plain, target, interest) → MLP → sigmoid
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

__all__ = ["MIMN"]


class MemoryNetwork(nn.Module):
    """Multi-slot memory for capturing user interest distribution.

    - K memory slots, each d-dimensional
    - Write: behavior sequence → attention over slots → slot update
    - Read: target query → attention over slots → weighted sum
    """

    def __init__(self, num_slots: int, emb_dim: int) -> None:
        super().__init__()
        self.num_slots = num_slots
        self.emb_dim = emb_dim
        self.memory = nn.Parameter(torch.zeros(1, num_slots, emb_dim))
        nn.init.xavier_uniform_(self.memory)

        self.write_proj = nn.Linear(emb_dim, emb_dim, bias=False)
        self.read_proj = nn.Linear(emb_dim, emb_dim, bias=False)

    def write(self, seq_embs: Tensor) -> None:
        """Write a sequence of behavior embeddings to memory.

        seq_embs: [B, seq_len, d]
        """
        B, T, d = seq_embs.shape
        keys = self.write_proj(seq_embs)                         # [B, T, d]
        scores = torch.bmm(keys, self.memory.expand(B, -1, -1).transpose(1, 2))  # [B, T, K]
        weights = torch.softmax(scores, dim=-1)                   # [B, T, K]
        weighted_seq = torch.bmm(weights.transpose(1, 2), seq_embs)  # [B, K, d]
        self.memory.data = self.memory.data + weighted_seq.mean(dim=0, keepdim=True)  # update

    def read(self, query: Tensor) -> Tensor:
        """Read interest vector from memory using target query.

        query: [B, d]
        return: [B, d]
        """
        B = query.size(0)
        q = self.read_proj(query).unsqueeze(1)                   # [B, 1, d]
        scores = torch.bmm(q, self.memory.expand(B, -1, -1).transpose(1, 2)).squeeze(1)  # [B, K]
        weights = torch.softmax(scores, dim=-1).unsqueeze(-1)     # [B, K, 1]
        return (self.memory.expand(B, -1, -1) * weights).sum(dim=1)  # [B, d]


class MIMN(BaseModel):
    """Multi-channel Interest with Moment Network for CTR prediction."""

    def __init__(self, model_cfg: DIENModelConfig) -> None:
        super().__init__()
        self._validate_fields(model_cfg)
        self.fields_cfg: Mapping[str, FieldEntry] = model_cfg.embedding_fields

        self.behavior_fields = model_cfg.behavior_fields
        self.target_fields = model_cfg.target_fields
        reserved = set(self.behavior_fields) | set(self.target_fields)
        self.field_names = [n for n in self.fields_cfg if n not in reserved]
        self.emb_size = int(self.fields_cfg[self.behavior_fields[0]].emb_size)

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

        # Behavior embeddings
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

        # Interest extractor config
        ie_cfg: dict[str, Any] = model_cfg.interest_extractor
        lstm_hidden = int(ie_cfg.get("lstm_hidden", 64))
        num_memory_slots = int(ie_cfg.get("num_memory_slots", 8))

        # Bi-LSTM for behavior sequence encoding
        self.bi_lstm = nn.LSTM(
            input_size=self.emb_size,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        # Memory network
        self.memory_net = MemoryNetwork(num_memory_slots, lstm_hidden * 2)

        # MLP
        plain_dim = sum(int(self.fields_cfg[fn].emb_size) for fn in self.field_names)
        target_dim = sum(int(self.fields_cfg[tf].emb_size) for tf in self.target_fields)
        mlp_input_dim = plain_dim + target_dim + lstm_hidden * 2

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

        self.interest_dim = lstm_hidden * 2
        self.target_proj = nn.Linear(self.emb_size, self.interest_dim, bias=False) if self.emb_size != self.interest_dim else nn.Identity()
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
        target_for_mem = (torch.stack(target_embs, dim=0).mean(dim=0) if target_embs
                          else torch.zeros(batch_size, self.emb_size, device=device))

        # 3. Behavior sequence → Bi-LSTM → interest states
        bf = self.behavior_fields[0]
        indices = to_device(feature_bags[bf]["indices"].long(), device)
        offsets = to_device(feature_bags[bf]["offsets"].long(), device)
        padded_ids, lengths, _ = bag_to_padded(indices, offsets)
        seq_emb = self.behavior_embeddings[bf](padded_ids)       # [B, T, d]
        lstm_out, _ = self.bi_lstm(seq_emb)                      # [B, T, 2h]

        # 4. Memory write: write LSTM states to memory
        self.memory_net.write(lstm_out)

        # 5. Memory read: target reads interest from memory
        target_query = self.target_proj(target_for_mem)           # [B, 2h]
        interest = self.memory_net.read(target_query)              # [B, 2h]

        # 6. Concat → MLP
        combined = torch.cat([plain_concat, target_concat, interest], dim=-1)
        hidden = self.mlp(combined)
        logit = self.head(hidden).squeeze(-1)
        return torch.sigmoid(logit)
