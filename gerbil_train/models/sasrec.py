"""SASRec (Self-Attentive Sequential Recommendation) for CTR prediction.

SASRec uses causal (left-to-right) Transformer self-attention to model
user behavior sequences. Unlike BST, it does not append the target item;
interest is extracted solely from past behavior via causal masking.

Reference: https://arxiv.org/abs/1808.09781 (ICDM 2018)
"""

from __future__ import annotations

from typing import Mapping, Any

import torch
from torch import Tensor, nn

from gerbil_train.config.model_config import DIENModelConfig, FieldEntry
from gerbil_train.utils.embedding import bag_to_padded, embed_one_field, to_device
from gerbil_train.models.layers import FullyConnectedLayer
from gerbil_train.models.base_model import BaseModel

__all__ = ["SASRec"]


class SASRec(BaseModel):
    def __init__(self, model_cfg: DIENModelConfig) -> None:
        super().__init__()
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

        # Behavior embeddings
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

        # SASRec config
        sasrec_cfg: dict[str, Any] = model_cfg.interest_extractor
        num_heads = int(sasrec_cfg["num_heads"])
        num_layers = int(sasrec_cfg["num_layers"])
        ffn_hidden = int(sasrec_cfg["ffn_hidden"])
        dropout = float(sasrec_cfg["dropout"])

        self.pos_embedding = nn.Embedding(500, self.emb_size)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.emb_size, nhead=num_heads, dim_feedforward=ffn_hidden,
            dropout=dropout, batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # MLP
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
        self._validate_fields(model_cfg)
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

        # 3. Behavior sequence
        bf = self.behavior_fields[0]
        padded_ids, padded_weights, lengths, max_seq_len = bag_to_padded(feature_bags[bf], device)
        seq_emb = self.behavior_embeddings[bf](padded_ids) * padded_weights.unsqueeze(-1)
        weight_sum = padded_weights.sum(dim=-1, keepdim=True).clamp(min=1e-8).unsqueeze(-1)
        seq_emb = seq_emb / weight_sum

        B, T, d = seq_emb.shape

        # 4. Positional encoding
        pos_ids = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)
        seq_emb = seq_emb + self.pos_embedding(pos_ids)

        # 5. Causal Transformer
        pad_mask = torch.arange(T, device=device).unsqueeze(0) >= lengths.unsqueeze(1)
        causal_mask = torch.triu(torch.full((T, T), True, device=device), diagonal=1)
        transformer_out = self.transformer(seq_emb, mask=causal_mask, src_key_padding_mask=pad_mask)

        # 6. Take output at last valid position as interest
        last_idx = (lengths - 1).clamp(min=0)
        all_pad = lengths == 0
        interest = transformer_out[torch.arange(B), last_idx]
        interest = torch.where(all_pad.unsqueeze(-1), torch.zeros_like(interest), interest)

        # 7. Concat -> MLP
        combined = torch.cat([plain_concat, target_concat, interest], dim=-1)
        hidden = self.mlp(combined)
        logit = self.head(hidden).squeeze(-1)
        return torch.sigmoid(logit)
