"""SDIM (Semantic Deep Interest Model) for CTR prediction.

SDIM learns a probabilistic semantic mask over behavior sequences, filtering
out target-irrelevant items via a Gumbel-Sigmoid reparameterized gate.

Reference: KDD 2022
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

__all__ = ["SDIM"]


def gumbel_sigmoid(logits: Tensor, tau: float = 1.0, hard: bool = False) -> Tensor:
    """Gumbel-Sigmoid relaxation for Bernoulli sampling.

    :param logits: [*] unscaled logits
    :param tau: temperature
    :param hard: if True, use straight-through estimator
    :return: [*] samples in [0, 1]
    """
    gumbel_noise = -(-torch.rand_like(logits).log()).log()
    y = (logits + gumbel_noise) / tau
    y_soft = torch.sigmoid(y)
    if hard:
        y_hard = (y_soft > 0.5).to(logits.dtype)
        return y_hard + y_soft.detach() - y_soft
    return y_soft


class SemanticMask(nn.Module):
    """Probabilistic semantic gate over behavior items."""

    def __init__(self, emb_dim: int, hidden_dims: list[int] = None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [64, 32]
        layers = []
        inp = emb_dim * 4
        for h in hidden_dims:
            layers.extend([nn.Linear(inp, h), nn.ReLU()])
            inp = h
        layers.append(nn.Linear(inp, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, target: Tensor, behavior: Tensor, gumbel_tau: float = 1.0, gumbel_hard: bool = False) -> Tensor:
        """Compute mask for each behavior item.

        :param target: [B, d]
        :param behavior: [B, T, d]
        :return: mask [B, T] in [0, 1]
        """
        target_tile = target.unsqueeze(1).expand_as(behavior)
        cat_feat = torch.cat([target_tile, behavior, target_tile * behavior, target_tile - behavior], dim=-1)
        logits = self.mlp(cat_feat).squeeze(-1)
        if self.training:
            return gumbel_sigmoid(logits, tau=gumbel_tau, hard=gumbel_hard)
        return torch.sigmoid(logits)


class SDIM(BaseModel):
    def __init__(self, model_cfg: DIENModelConfig) -> None:
        super().__init__()
        self._validate_fields(model_cfg)
        self.embedding_fields: Mapping[str, FieldEntry] = model_cfg.embedding_fields

        self.behavior_fields = model_cfg.behavior_fields
        self.target_fields = model_cfg.target_fields
        reserved = set(self.behavior_fields) | set(self.target_fields)
        self.field_names = [n for n in self.embedding_fields if n not in reserved]
        self.emb_size = int(self.embedding_fields[self.behavior_fields[0]].emb_size)

        self.target_embedding_bags = nn.ModuleDict()
        for f_name in self.target_fields:
            entry = self.embedding_fields[f_name]
            key = str(entry.field_index)
            if key not in self.target_embedding_bags:
                self.target_embedding_bags[key] = nn.EmbeddingBag(
                    num_embeddings=int(entry.dim), embedding_dim=int(entry.emb_size), mode="sum",
                )

        self.behavior_embeddings = nn.ModuleDict()
        for bf in self.behavior_fields:
            entry = self.embedding_fields[bf]
            self.behavior_embeddings[bf] = nn.Embedding(
                num_embeddings=int(entry.dim) + 1, embedding_dim=int(entry.emb_size), padding_idx=int(entry.dim),
            )

        self.field_embedding_bags = nn.ModuleDict()
        for field_name in self.field_names:
            entry = self.embedding_fields[field_name]
            key = str(entry.field_index)
            if key not in self.field_embedding_bags:
                self.field_embedding_bags[key] = nn.EmbeddingBag(
                    num_embeddings=int(entry.dim), embedding_dim=int(entry.emb_size), mode="sum",
                )

        sdim_cfg: dict[str, Any] = model_cfg.interest_extractor
        mask_hidden = list(sdim_cfg["mask_hidden"])
        self.gumbel_tau = float(sdim_cfg["gumbel_tau"])
        self.gumbel_hard = bool(sdim_cfg["gumbel_hard"])

        self.semantic_mask = SemanticMask(self.emb_size, hidden_dims=mask_hidden)

        plain_dim = sum(int(self.embedding_fields[fn].emb_size) for fn in self.field_names)
        target_dim = sum(int(self.embedding_fields[tf].emb_size) for tf in self.target_fields)
        mlp_input_dim = plain_dim + target_dim + self.emb_size

        mlp_cfg: dict[str, Any] = model_cfg.mlp
        hidden_dims = list(mlp_cfg["hidden_dims"])
        self.mlp = FullyConnectedLayer(
            input_dim=mlp_input_dim, hidden_dims=hidden_dims, bias=[True] * len(hidden_dims),
            batch_norm=bool(mlp_cfg["batch_norm"]), activation=str(mlp_cfg["activation"]),
            dropout=float(mlp_cfg["dropout"]),
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
        target_for_sdim = (torch.stack(target_embs, dim=0).mean(dim=0) if target_embs
                           else torch.zeros(batch_size, self.emb_size, device=device))

        bf = self.behavior_fields[0]
        padded_ids, padded_weights, lengths, max_seq_len = bag_to_padded(feature_bags[bf], device)
        seq_emb = self.behavior_embeddings[bf](padded_ids) * padded_weights.unsqueeze(-1)
        weight_sum = padded_weights.sum(dim=-1, keepdim=True).clamp(min=1e-8).unsqueeze(-1)
        seq_emb = seq_emb / weight_sum

        B, T, d = seq_emb.shape

        # Semantic mask: gate irrelevant items
        mask = self.semantic_mask(target_for_sdim, seq_emb, gumbel_tau=self.gumbel_tau, gumbel_hard=self.gumbel_hard)

        # Apply padding mask
        pad_mask = torch.arange(T, device=device).unsqueeze(0) >= lengths.unsqueeze(1)
        mask = mask.masked_fill(pad_mask, 0.0)

        # Weighted aggregation
        interest = (seq_emb * mask.unsqueeze(-1)).sum(dim=1)
        mask_sum = mask.sum(dim=-1, keepdim=True).clamp(min=1)
        interest = interest / mask_sum

        combined = torch.cat([plain_concat, target_concat, interest], dim=-1)
        hidden = self.mlp(combined)
        logit = self.head(hidden).squeeze(-1)
        return torch.sigmoid(logit)
