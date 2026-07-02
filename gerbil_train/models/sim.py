"""SIM (Search-based Interest Model) for CTR prediction.

SIM addresses long-term user behavior modeling via a two-stage approach:
  1. GSU (General Search Unit): efficiently retrieves top-K relevant behaviors
  2. ESU (Exact Search Unit): multi-head attention on retrieved behaviors

This implementation uses soft-search (embedding similarity) for GSU.
"""

from __future__ import annotations

from typing import Mapping, Any

import torch
from torch import Tensor, nn

from gerbil_train.config.model_config import DIENModelConfig, FieldEntry
from gerbil_train.utils.embedding import bag_to_padded, embed_one_field, to_device
from gerbil_train.models.layers import FullyConnectedLayer
from gerbil_train.models.base_model import BaseModel

__all__ = ["SIM"]


class GSU(nn.Module):
    """General Search Unit: retrieves top-K items from behavior sequence."""

    def __init__(self, emb_dim: int, top_k: int):
        super().__init__()
        self.top_k = top_k
        self.query_proj = nn.Linear(emb_dim, emb_dim, bias=False)

    def forward(self, seq_emb: Tensor, target_emb: Tensor, lengths: Tensor) -> tuple[Tensor, Tensor]:
        """Retrieve top-K relevant items.

        :param seq_emb: [B, T, d] behavior embeddings
        :param target_emb: [B, d] target embedding
        :param lengths: [B] actual sequence lengths
        :return: (top_k_emb [B, K, d], mask [B, K])
        """
        B, T, d = seq_emb.shape
        K = min(self.top_k, T)
        query = self.query_proj(target_emb).unsqueeze(1)
        scores = torch.bmm(seq_emb, query.transpose(1, 2)).squeeze(-1)
        mask = torch.arange(T, device=scores.device).unsqueeze(0) >= lengths.unsqueeze(1)
        scores = scores.masked_fill(mask, float("-inf"))
        top_scores, top_idx = scores.topk(K, dim=1)
        top_emb = torch.gather(seq_emb, 1, top_idx.unsqueeze(-1).expand(-1, -1, d))
        return top_emb, top_scores


class ESU(nn.Module):
    """Exact Search Unit: multi-head attention over retrieved items."""

    def __init__(self, emb_dim: int, num_heads: int = 4):
        super().__init__()
        self.mha = nn.MultiheadAttention(emb_dim, num_heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(emb_dim, emb_dim * 2),
            nn.ReLU(),
            nn.Linear(emb_dim * 2, emb_dim),
        )
        self.norm = nn.LayerNorm(emb_dim)

    def forward(self, top_emb: Tensor, target_emb: Tensor, top_scores: Tensor) -> Tensor:
        query = target_emb.unsqueeze(1)
        attn_out, _ = self.mha(query, top_emb, top_emb)
        attn_out = self.norm(query + attn_out)
        ffn_out = self.ffn(attn_out)
        return self.norm(attn_out + ffn_out).squeeze(1)


class SIM(BaseModel):
    """Search-based Interest Model for CTR prediction."""

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

        # SIM config
        sim_cfg: dict[str, Any] = model_cfg.interest_extractor
        top_k = int(sim_cfg.get("top_k", 20))
        esu_heads = int(sim_cfg.get("esu_heads", 4))

        self.gsu = GSU(self.emb_size, top_k)
        self.esu = ESU(self.emb_size, esu_heads)

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
        target_for_sim = (torch.stack(target_embs, dim=0).mean(dim=0) if target_embs
                          else torch.zeros(batch_size, self.emb_size, device=device))

        # 3. Behavior sequence
        bf = self.behavior_fields[0]
        indices = to_device(feature_bags[bf]["indices"].long(), device)
        offsets = to_device(feature_bags[bf]["offsets"].long(), device)
        padded_ids, lengths, _ = bag_to_padded(indices, offsets)
        seq_emb = self.behavior_embeddings[bf](padded_ids)

        # 4. GSU: retrieve top-K
        top_emb, top_scores = self.gsu(seq_emb, target_for_sim, lengths)

        # 5. ESU: multi-head attention on top-K
        interest = self.esu(top_emb, target_for_sim, top_scores)

        # 6. Concat -> MLP
        combined = torch.cat([plain_concat, target_concat, interest], dim=-1)
        hidden = self.mlp(combined)
        logit = self.head(hidden).squeeze(-1)
        return torch.sigmoid(logit)
