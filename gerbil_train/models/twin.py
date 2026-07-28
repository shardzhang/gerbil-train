"""TWIN (Two-stage Interest Discovery Network) for CTR prediction.

TWIN discovers multiple user interests via a two-stage approach:
  Stage 1 — Interest Discovery: cluster behavior items into K interest vectors
  Stage 2 — Interest Aggregation: target-aware attention over the K interests

Reference: https://arxiv.org/abs/2302.09894 (2023)
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

__all__ = ["TWIN"]


class InterestDiscovery(nn.Module):
    """Stage 1: discover K interest vectors from behavior sequence via prototype assignment."""

    def __init__(self, emb_dim: int, num_interests: int):
        super().__init__()
        self.num_interests = num_interests
        # Learnable interest prototypes
        self.prototypes = nn.Parameter(torch.randn(num_interests, emb_dim) * 0.1)

    def forward(self, seq_emb: Tensor, mask: Tensor) -> Tensor:
        """Discover interests.

        :param seq_emb: [B, T, d] behavior embeddings
        :param mask: [B, T] True for padding
        :return: [B, K, d] K interest vectors
        """
        # Assignment scores: each item → each prototype
        # [B, T, K] — soft assignment based on similarity
        scores = torch.bmm(seq_emb, self.prototypes.T.unsqueeze(0).expand(seq_emb.size(0), -1, -1))
        scores = scores.masked_fill(mask.unsqueeze(-1), float("-inf"))
        assignment = F.softmax(scores, dim=-1)                    # [B, T, K]

        # Aggregate items per interest: weighted sum over T
        # [B, T, K, 1] x [B, T, 1, d] -> [B, K, d]
        interests = (assignment.unsqueeze(-1) * seq_emb.unsqueeze(2)).sum(dim=1)

        # Normalize each interest vector
        interest_norm = interests.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        interests = interests / interest_norm
        return interests


class InterestAggregation(nn.Module):
    """Stage 2: target-aware attention over discovered interests."""

    def __init__(self, emb_dim: int, num_interests: int):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(emb_dim * 2, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, 1),
        )

    def forward(self, interests: Tensor, target_emb: Tensor) -> Tensor:
        """Attend over interests guided by target.

        :param interests: [B, K, d]
        :param target_emb: [B, d]
        :return: [B, d] aggregated interest
        """
        target_tile = target_emb.unsqueeze(1).expand(-1, interests.size(1), -1)
        pair = torch.cat([interests, target_tile], dim=-1)
        scores = self.attn(pair).squeeze(-1)                      # [B, K]
        attn = F.softmax(scores, dim=-1)
        return (attn.unsqueeze(-1) * interests).sum(dim=1)


class TWIN(BaseModel):
    """Two-stage Interest Discovery Network for CTR prediction."""

    def __init__(self, model_cfg: DIENModelConfig) -> None:
        super().__init__()
        self._validate_fields(model_cfg)
        self.embedding_fields: Mapping[str, FieldEntry] = model_cfg.embedding_fields

        self.behavior_fields = model_cfg.behavior_fields
        self.target_fields = model_cfg.target_fields
        reserved = set(self.behavior_fields) | set(self.target_fields)
        self.field_names = [n for n in self.embedding_fields if n not in reserved]
        self.emb_size = int(self.embedding_fields[self.behavior_fields[0]].emb_size)

        # Target embeddings
        self.target_embedding_bags = nn.ModuleDict()
        for f_name in self.target_fields:
            entry = self.embedding_fields[f_name]
            key = str(entry.field_index)
            if key not in self.target_embedding_bags:
                self.target_embedding_bags[key] = nn.EmbeddingBag(
                    num_embeddings=int(entry.dim), embedding_dim=int(entry.emb_size), mode="sum",
                )

        # Behavior sequence embeddings
        self.behavior_embeddings = nn.ModuleDict()
        for bf in self.behavior_fields:
            entry = self.embedding_fields[bf]
            self.behavior_embeddings[bf] = nn.Embedding(
                num_embeddings=int(entry.dim) + 1, embedding_dim=int(entry.emb_size), padding_idx=int(entry.dim),
            )

        # Plain field embeddings
        self.field_embedding_bags = nn.ModuleDict()
        for field_name in self.field_names:
            entry = self.embedding_fields[field_name]
            key = str(entry.field_index)
            if key not in self.field_embedding_bags:
                self.field_embedding_bags[key] = nn.EmbeddingBag(
                    num_embeddings=int(entry.dim), embedding_dim=int(entry.emb_size), mode="sum",
                )

        # TWIN config
        twin_cfg: dict[str, Any] = model_cfg.interest_extractor
        num_interests = int(twin_cfg.get("num_interests", 4))

        self.interest_discovery = InterestDiscovery(self.emb_size, num_interests)
        self.interest_aggregation = InterestAggregation(self.emb_size, num_interests)

        # MLP
        plain_dim = sum(int(self.embedding_fields[fn].emb_size) for fn in self.field_names)
        target_dim = sum(int(self.embedding_fields[tf].emb_size) for tf in self.target_fields)
        mlp_input_dim = plain_dim + target_dim + self.emb_size

        mlp_cfg: dict[str, Any] = model_cfg.mlp
        hidden_dims = list(mlp_cfg.get("hidden_dims", [256, 128]))
        self.mlp = FullyConnectedLayer(
            input_dim=mlp_input_dim, hidden_dims=hidden_dims, bias=[True] * len(hidden_dims),
            batch_norm=bool(mlp_cfg.get("batch_norm", False)), activation=str(mlp_cfg.get("activation", "relu")),
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
            entry = self.embedding_fields[fn]
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
            entry = self.embedding_fields[tf]
            emb = embed_one_field(
                self.target_embedding_bags[str(entry.field_index)],
                feature_bags[tf]["indices"], feature_bags[tf]["offsets"],
                feature_bags[tf]["weights"], device=device,
            )
            target_embs.append(emb)
        target_concat = torch.cat(target_embs, dim=-1) if target_embs else torch.zeros(batch_size, 0, device=device)
        target_for_twin = (torch.stack(target_embs, dim=0).mean(dim=0) if target_embs
                           else torch.zeros(batch_size, self.emb_size, device=device))

        # 3. Behavior sequence
        bf = self.behavior_fields[0]
        indices = to_device(feature_bags[bf]["indices"].long(), device)
        offsets = to_device(feature_bags[bf]["offsets"].long(), device)
        padded_ids, padded_weights, lengths, _ = bag_to_padded(feature_bags[bf], device)
        seq_emb = self.behavior_embeddings[bf](padded_ids) * padded_weights.unsqueeze(-1)
        weight_sum = padded_weights.sum(dim=-1, keepdim=True).clamp(min=1e-8).unsqueeze(-1)
        seq_emb = seq_emb / weight_sum

        # 4. Stage 1: Interest Discovery
        B, T, d = seq_emb.shape
        pad_mask = torch.arange(T, device=device).unsqueeze(0) >= lengths.unsqueeze(1)
        interests = self.interest_discovery(seq_emb, pad_mask)

        # 5. Stage 2: Target-aware Interest Aggregation
        interest = self.interest_aggregation(interests, target_for_twin)

        # 6. Concat → MLP
        combined = torch.cat([plain_concat, target_concat, interest], dim=-1)
        hidden = self.mlp(combined)
        logit = self.head(hidden).squeeze(-1)
        return torch.sigmoid(logit)
