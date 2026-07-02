"""MIND (Multi-Interest Network with Dynamic Routing) for CTR prediction.

MIND extracts multiple interest vectors from user behavior sequences via
capsule-based dynamic routing (B2I routing), then uses label-aware attention
to select the most relevant interest for each target item.

Reference: https://arxiv.org/abs/1904.08030 (CIKM 2019)
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

__all__ = ["MIND"]


class CapsuleLayer(nn.Module):
    """Dynamic routing layer for extracting interest capsules from behavior sequences."""

    def __init__(self, input_dim: int, output_dim: int, num_capsules: int, num_iterations: int = 3):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_capsules = num_capsules
        self.num_iterations = num_iterations
        self.bilinear = nn.Linear(input_dim, output_dim * num_capsules, bias=False)

    def _squash(self, x: Tensor) -> Tensor:
        norm = x.norm(dim=-1, keepdim=True)
        scale = norm / (1 + norm ** 2)
        return scale * x

    def forward(self, behavior_emb: Tensor, mask: Tensor) -> Tensor:
        """Extract interest capsules via dynamic routing.

        :param behavior_emb: [B, T, d] behavior embeddings
        :param mask: [B, T] True for padding positions (masked out)
        :return: [B, K, d] interest capsules
        """
        B, T, d = behavior_emb.shape
        K = self.num_capsules

        # [B, T, K*d'] -> [B, T, K, d']
        u_hat = self.bilinear(behavior_emb).view(B, T, K, self.output_dim)

        # mask: [B, T, 1, 1]
        mask_expanded = mask.unsqueeze(-1).unsqueeze(-1)

        b = behavior_emb.new_zeros(B, T, K)

        # [B, 1, 1] flag for rows where all positions are masked
        all_masked = mask.all(dim=1, keepdim=True).unsqueeze(-1)

        for _ in range(self.num_iterations):
            b_masked = b.masked_fill(mask.unsqueeze(-1), float("-inf"))
            b_safe = torch.where(all_masked, torch.zeros_like(b), b_masked)
            w = F.softmax(b_safe, dim=1)
            z = (w.unsqueeze(-1) * u_hat).sum(dim=1)
            v = self._squash(z)
            if _ < self.num_iterations - 1:
                b = b + (u_hat * v.unsqueeze(1)).sum(dim=-1)

        return v


class LabelAwareAttention(nn.Module):
    """Attention that selects the most relevant interest capsule for the target item."""

    def __init__(self, embed_dim: int):
        super().__init__()

    def forward(self, target_emb: Tensor, interests: Tensor) -> Tensor:
        """Attend over interest capsules guided by target embedding.

        :param target_emb: [B, d] target embedding
        :param interests: [B, K, d] interest capsules
        :return: [B, d] aggregated interest
        """
        scores = torch.bmm(interests, target_emb.unsqueeze(-1)).squeeze(-1)
        attn = F.softmax(scores, dim=-1)
        return (attn.unsqueeze(-1) * interests).sum(dim=1)


class MIND(BaseModel):
    """Multi-Interest Network with Dynamic Routing for CTR prediction."""

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

        # MIND config
        mind_cfg: dict[str, Any] = model_cfg.interest_extractor
        num_interests = int(mind_cfg.get("num_interests", 4))
        routing_iters = int(mind_cfg.get("routing_iters", 3))

        self.capsule_layer = CapsuleLayer(
            input_dim=self.emb_size, output_dim=self.emb_size,
            num_capsules=num_interests, num_iterations=routing_iters,
        )
        self.label_attention = LabelAwareAttention(self.emb_size)

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
        target_for_mind = (torch.stack(target_embs, dim=0).mean(dim=0) if target_embs
                           else torch.zeros(batch_size, self.emb_size, device=device))

        # 3. Behavior sequence
        bf = self.behavior_fields[0]
        indices = to_device(feature_bags[bf]["indices"].long(), device)
        offsets = to_device(feature_bags[bf]["offsets"].long(), device)
        padded_ids, lengths, _ = bag_to_padded(indices, offsets)
        seq_emb = self.behavior_embeddings[bf](padded_ids)

        # 4. Dynamic routing: extract multi-interest capsules
        T = seq_emb.size(1)
        mask = torch.arange(T, device=device).unsqueeze(0) >= lengths.unsqueeze(1)
        interests = self.capsule_layer(seq_emb, mask)

        # 5. Label-aware attention: select relevant interest for target
        interest = self.label_attention(target_for_mind, interests)

        # 6. Concat -> MLP
        combined = torch.cat([plain_concat, target_concat, interest], dim=-1)
        hidden = self.mlp(combined)
        logit = self.head(hidden).squeeze(-1)
        return torch.sigmoid(logit)
