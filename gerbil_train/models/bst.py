"""BST (Behavior Sequence Transformer) for CTR prediction.

BST applies Transformer encoder to user behavior sequences, modeling
complex pairwise item-item interactions for interest extraction.

Reference: https://arxiv.org/abs/1905.06874 (KDD 2019)
"""

from __future__ import annotations

from typing import Mapping, Any

import torch
from torch import Tensor, nn

from gerbil_train.config.model_config import DIENModelConfig, FieldEntry
from gerbil_train.utils.embedding import bag_to_padded, embed_one_field, to_device
from gerbil_train.models.layers import FullyConnectedLayer
from gerbil_train.models.base_model import BaseModel

__all__ = ["BST"]


class BST(BaseModel):
    """Behavior Sequence Transformer for CTR prediction."""

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

        # BST config
        bst_cfg: dict[str, Any] = model_cfg.interest_extractor
        num_heads = int(bst_cfg.get("num_heads", 4))
        num_layers = int(bst_cfg.get("num_layers", 2))
        ffn_hidden = int(bst_cfg.get("ffn_hidden", self.emb_size * 2))
        dropout = float(bst_cfg.get("dropout", 0.1))

        self.pos_embedding = nn.Embedding(num_embeddings=500, embedding_dim=self.emb_size)

        transformer_layer = nn.TransformerEncoderLayer(
            d_model=self.emb_size, nhead=num_heads, dim_feedforward=ffn_hidden,
            dropout=dropout, batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(transformer_layer, num_layers=num_layers)

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
        dtype = next(self.parameters()).dtype

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
        plain_concat = torch.cat(plain_embs, dim=-1) if plain_embs else torch.zeros(batch_size, 0, device=device, dtype=dtype)

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
        target_concat = torch.cat(target_embs, dim=-1) if target_embs else torch.zeros(batch_size, 0, device=device, dtype=dtype)
        target_for_bst = (torch.stack(target_embs, dim=0).mean(dim=0) if target_embs
                          else torch.zeros(batch_size, self.emb_size, device=device, dtype=dtype))

        # 3. Behavior sequence
        bf = self.behavior_fields[0]
        indices = to_device(feature_bags[bf]["indices"].long(), device)
        offsets = to_device(feature_bags[bf]["offsets"].long(), device)
        padded_ids, lengths, _ = bag_to_padded(indices, offsets)
        seq_emb = self.behavior_embeddings[bf](padded_ids)

        # 4. Append target to sequence
        B, T, d = seq_emb.shape
        target_tile = target_for_bst.unsqueeze(1)
        combined_seq = torch.cat([seq_emb, target_tile], dim=1)
        seq_len = lengths + 1

        # 5. Positional encoding
        max_len = combined_seq.size(1)
        pos_ids = torch.arange(max_len, device=device).unsqueeze(0).expand(B, -1)
        pos_emb = self.pos_embedding(pos_ids)
        combined_seq = combined_seq + pos_emb

        # 6. Transformer encoder (key_padding_mask=True for padding)
        pad_mask = torch.arange(max_len, device=device).unsqueeze(0) >= seq_len.unsqueeze(1)
        transformer_out = self.transformer(combined_seq, src_key_padding_mask=pad_mask)

        # 7. Take output at target position (last non-padded position)
        interest = transformer_out[torch.arange(B), seq_len - 1]

        # 8. Concat -> MLP
        combined = torch.cat([plain_concat, target_concat, interest], dim=-1)
        hidden = self.mlp(combined)
        logit = self.head(hidden).squeeze(-1)
        return torch.sigmoid(logit)
