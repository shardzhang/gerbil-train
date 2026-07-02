"""AFM (Attentional Factorization Machine) for CTR prediction.

AFM = Linear (1st-order) + Attentional FM (2nd-order pair-wise with attention).

The key difference from FM is that each pair-wise interaction (v_i ⊙ v_j) is
weighted by a learned attention score before summation.
"""

from __future__ import annotations

from typing import Mapping

import torch
from torch import Tensor, nn

from gerbil_train.config.model_config import BaseModelConfig, FieldEntry
from gerbil_train.utils.embedding import embed_one_field
from gerbil_train.models.base_model import BaseModel

__all__ = ["AFM"]


class AFM(BaseModel):
    """Attentional Factorization Machine for CTR prediction.

    AFM enhances FM by learning an attention weight for each pair-wise
    feature interaction via a small MLP.
    """

    def __init__(self, model_cfg: BaseModelConfig) -> None:
        super().__init__()
        self._validate_fields(model_cfg)

        self.fields_cfg: Mapping[str, FieldEntry] = model_cfg.embedding_fields
        self.field_names = list(self.fields_cfg.keys())
        self.num_fields = len(self.field_names)
        self.emb_size = int(next(iter(self.fields_cfg.values())).emb_size)

        # Linear embeddings: vocab → 1
        self.linear_embeddings = nn.ModuleDict()
        # Feature embeddings: vocab → k
        self.afm_embeddings = nn.ModuleDict()

        for field_name, entry in self.fields_cfg.items():
            if entry.field_type == 0 and entry.concat_type == "direct":
                continue
            key = str(entry.field_index)
            if key not in self.linear_embeddings:
                self.linear_embeddings[key] = nn.EmbeddingBag(
                    num_embeddings=int(entry.dim),
                    embedding_dim=1,
                    mode="sum",
                )
            if key not in self.afm_embeddings:
                self.afm_embeddings[key] = nn.EmbeddingBag(
                    num_embeddings=int(entry.dim),
                    embedding_dim=int(self.emb_size),
                    mode="sum",
                )

        # Attention network: takes paired embedding (v_i ⊙ v_j) → scalar score
        attn_hidden = model_cfg.afm_attention.get("hidden_size", 128)
        attn_dropout = model_cfg.afm_attention.get("dropout", 0.0)
        self.attention = nn.Sequential(
            nn.Linear(self.emb_size, attn_hidden),
            nn.ReLU(),
            nn.Dropout(attn_dropout),
            nn.Linear(attn_hidden, 1),
        )

        # Final prediction layer on the attention-weighted sum of pairs
        self.pairwise_head = nn.Linear(self.emb_size, 1, bias=False)
        self.bias = nn.Parameter(torch.zeros(1))
        self.reset_parameters()

    def _validate_fields(self, model_cfg: BaseModelConfig) -> None:
        if not model_cfg.embedding_fields:
            raise ValueError("embedding_fields must be a non-empty mapping")
        emb_sizes = {int(e.emb_size) for e in model_cfg.embedding_fields.values()
                     if not (e.field_type == 0 and e.concat_type == "direct")}
        if len(emb_sizes) > 1:
            raise ValueError(f"AFM requires all field embeddings to have the same size, got {emb_sizes}")

    def reset_parameters(self) -> None:
        nn.init.zeros_(self.bias)
        for emb in self.linear_embeddings.values():
            nn.init.xavier_uniform_(emb.weight)
        for emb in self.afm_embeddings.values():
            nn.init.xavier_uniform_(emb.weight)
        if self.pairwise_head is not None:
            nn.init.xavier_uniform_(self.pairwise_head.weight)
            if self.pairwise_head.bias is not None:
                nn.init.zeros_(self.pairwise_head.bias)

    def forward(self, feature_bags: Mapping[str, Mapping[str, Tensor]]) -> Tensor:
        first_offsets = feature_bags[self.field_names[0]]["offsets"]
        batch_size = int(first_offsets.size(0))
        device = next(self.parameters()).device

        # 1. Linear term: w_0 + Σ w_i · x_i
        linear_sum = self.bias.expand(batch_size).to(device)
        for field_name, entry in self.fields_cfg.items():
            if entry.field_type == 0 and entry.concat_type == "direct":
                continue
            linear_emb = embed_one_field(
                self.linear_embeddings[str(entry.field_index)],
                feature_bags[field_name]["indices"],
                feature_bags[field_name]["offsets"],
                feature_bags[field_name]["weights"],
                device=device,
            )
            linear_sum = linear_sum + linear_emb.squeeze(-1)

        # 2. Collect field embeddings (exclude direct fields)
        fm_emb_list: list[Tensor] = []
        for field_name, entry in self.fields_cfg.items():
            if entry.field_type == 0 and entry.concat_type == "direct":
                continue
            feature_emb = embed_one_field(
                self.afm_embeddings[str(entry.field_index)],
                feature_bags[field_name]["indices"],
                feature_bags[field_name]["offsets"],
                feature_bags[field_name]["weights"],
                device=device,
            )
            fm_emb_list.append(feature_emb)

        # 3. Compute all pair-wise interactions (v_i ⊙ v_j)
        stacked = torch.stack(fm_emb_list, dim=1)        # [B, n, k]
        n = stacked.size(1)
        row, col = [], []
        for i in range(n):
            for j in range(i + 1, n):
                row.append(i)
                col.append(j)

        if not row:  # fewer than 2 fields, skip FM term
            attn_fm_term = torch.zeros(batch_size, device=device)
        else:
            idx_i = torch.tensor(row, device=device)
            idx_j = torch.tensor(col, device=device)
            pairs = stacked[:, idx_i, :] * stacked[:, idx_j, :]       # [B, n_pairs, k]

            # Attention scores: [B, n_pairs, 1] → softmax over pairs
            attn_scores = self.attention(pairs)                       # [B, n_pairs, 1]
            attn_weights = torch.softmax(attn_scores, dim=1)           # [B, n_pairs, 1]

            # Weighted sum over pairs: [B, k]
            attn_pooled = (attn_weights * pairs).sum(dim=1)            # [B, k]
            attn_fm_term = self.pairwise_head(attn_pooled).squeeze(-1)  # [B]

        logits = linear_sum + attn_fm_term
        return torch.sigmoid(logits)
