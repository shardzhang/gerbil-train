"""ETA (End-to-end Target Attention) for CTR prediction.

ETA uses locality-sensitive hashing (LSH) to constrain target attention to
only behavior items sharing a hash bucket with the target, enabling efficient
long-sequence modeling.

Reference: https://dl.acm.org/doi/10.1145/3459637.3482270 (CIKM 2021)
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

__all__ = ["ETA"]


class HashEncoder(nn.Module):
    """Multi-table hash encoder with straight-through gradient estimator.

    Projects input embeddings into m binary hash codes, each of k bits.
    During training, uses straight-through estimator for differentiable hashing.
    """

    def __init__(self, input_dim: int, num_tables: int = 4, num_bits: int = 4):
        super().__init__()
        self.num_tables = num_tables
        self.num_bits = num_bits
        self.projections = nn.Parameter(torch.randn(num_tables, input_dim, num_bits) * 0.1)

    def forward(self, x: Tensor) -> Tensor:
        """Hash input to binary codes.

        :param x: [*, input_dim]
        :return: [*, num_tables, num_bits] with values in {-1, 1}
        """
        scores = torch.einsum("...d,tdu->...tu", x, self.projections)
        bits = torch.sign(scores)
        if self.training:
            bits = scores + (bits - scores).detach()
        return bits


class ETA(BaseModel):
    """End-to-end Target Attention for CTR prediction."""

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
                    num_embeddings=int(entry.dim), embedding_dim=int(entry.emb_size), mode="sum",
                )

        # Behavior embeddings
        self.behavior_embeddings = nn.ModuleDict()
        for bf in self.behavior_fields:
            entry = self.fields_cfg[bf]
            self.behavior_embeddings[bf] = nn.Embedding(
                num_embeddings=int(entry.dim) + 1, embedding_dim=int(entry.emb_size), padding_idx=int(entry.dim),
            )

        # Plain field embeddings
        self.field_embedding_bags = nn.ModuleDict()
        for field_name in self.field_names:
            entry = self.fields_cfg[field_name]
            key = str(entry.field_index)
            if key not in self.field_embedding_bags:
                self.field_embedding_bags[key] = nn.EmbeddingBag(
                    num_embeddings=int(entry.dim), embedding_dim=int(entry.emb_size), mode="sum",
                )

        # ETA config
        eta_cfg: dict[str, Any] = model_cfg.interest_extractor
        num_tables = int(eta_cfg.get("num_tables", 4))
        num_bits = int(eta_cfg.get("num_bits", 4))

        self.hash_encoder = HashEncoder(self.emb_size, num_tables=num_tables, num_bits=num_bits)

        # Activation Unit (same as DIN)
        au_hidden = dict(model_cfg.local_activation_unit).get("hidden_dims", [32, 16])
        self.activation_unit = nn.Sequential(
            nn.Linear(self.emb_size * 3, int(au_hidden[0])), nn.ReLU(),
            nn.Linear(int(au_hidden[0]), int(au_hidden[1])), nn.ReLU(),
            nn.Linear(int(au_hidden[1]), 1),
        )

        # MLP
        plain_dim = sum(int(self.fields_cfg[fn].emb_size) for fn in self.field_names)
        target_dim = sum(int(self.fields_cfg[tf].emb_size) for tf in self.target_fields)
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
        target_for_eta = (torch.stack(target_embs, dim=0).mean(dim=0) if target_embs
                          else torch.zeros(batch_size, self.emb_size, device=device))

        # 3. Behavior sequence
        bf = self.behavior_fields[0]
        indices = to_device(feature_bags[bf]["indices"].long(), device)
        offsets = to_device(feature_bags[bf]["offsets"].long(), device)
        padded_ids, lengths, _ = bag_to_padded(indices, offsets)
        seq_emb = self.behavior_embeddings[bf](padded_ids)

        # 4. Hash-constrained attention
        B, T, d = seq_emb.shape
        pad_mask = torch.arange(T, device=device).unsqueeze(0) >= lengths.unsqueeze(1)

        # Hash behavior items and target
        seq_hash = self.hash_encoder(seq_emb)
        tgt_hash = self.hash_encoder(target_for_eta)

        # Hash match: items sharing at least one complete table match with target
        match = (seq_hash == tgt_hash.unsqueeze(1)).all(dim=-1)
        any_match = match.any(dim=-1)

        # Activation unit: DIN-style attention
        target_tile = target_for_eta.unsqueeze(1).expand(-1, T, -1)
        cat_emb = torch.cat([seq_emb, target_tile, seq_emb * target_tile], dim=-1)
        raw_scores = self.activation_unit(cat_emb).squeeze(-1)

        # Mask: padding OR no hash match
        attention_mask = pad_mask | ~any_match
        scores = raw_scores.masked_fill(attention_mask, float("-inf"))
        scores_safe = torch.where(attention_mask.all(dim=-1, keepdim=True),
                                  torch.zeros_like(scores), scores)
        attn = F.softmax(scores_safe, dim=-1)

        interest = (attn.unsqueeze(-1) * seq_emb).sum(dim=1)

        # 5. Concat -> MLP
        combined = torch.cat([plain_concat, target_concat, interest], dim=-1)
        hidden = self.mlp(combined)
        logit = self.head(hidden).squeeze(-1)
        return torch.sigmoid(logit)
