"""MaskNet for CTR prediction.

MaskNet uses instance-guided mask blocks: each block generates a sample-
dependent mask from the input features, applies it element-wise, then
passes through a FC layer. Multiple blocks can be stacked (Serial MaskNet).

Reference: https://arxiv.org/abs/2102.07619 (2021)
"""

from __future__ import annotations

from typing import Mapping, Any

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from gerbil_train.config.model_config import BaseModelConfig, FieldEntry
from gerbil_train.utils.embedding import embed_one_field
from gerbil_train.models.base_model import BaseModel

__all__ = ["MaskNet"]


class _InstanceMask(nn.Module):
    """Instance-guided mask generation.

    Mask = ReLU(W_2 · LayerNorm(sum_pool(E)) + b_2)
    where E is the concat of all field embeddings.
    """

    def __init__(self, num_fields: int, field_emb_size: int, hidden_dim: int):
        super().__init__()
        self.mask_gen = nn.Sequential(
            nn.Linear(field_emb_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_fields * field_emb_size),
        )
        self.field_emb_size = field_emb_size

    def forward(self, field_embs: list[Tensor]) -> Tensor:
        """Generate mask from field embeddings.

        :param field_embs: list of [B, d] field embeddings
        :return: mask [B, N*d] (instance-dependent)
        """
        stacked = torch.stack(field_embs, dim=1)    # [B, N, d]
        pooled = stacked.mean(dim=1)                 # [B, d]
        return self.mask_gen(pooled)                 # [B, N*d]


class _MaskBlock(nn.Module):
    """One MaskNet block: instance mask → element-wise multiply → FC."""

    def __init__(self, num_fields: int, field_emb_size: int, hidden_dim: int):
        super().__init__()
        self.instance_mask = _InstanceMask(num_fields, field_emb_size, hidden_dim)
        self.fc = nn.Sequential(
            nn.Linear(num_fields * field_emb_size, hidden_dim),
            nn.ReLU(),
        )

    def forward(self, field_embs: list[Tensor]) -> Tensor:
        concat = torch.cat(field_embs, dim=-1)       # [B, N*d]
        mask = self.instance_mask(field_embs)         # [B, N*d]
        masked = concat * mask
        return self.fc(masked)                        # [B, h]


class _SerialMaskBlock(nn.Module):
    """Mask block for serial connection: projects hidden back to input dim."""

    def __init__(self, num_fields: int, field_emb_size: int, input_dim: int, hidden_dim: int):
        super().__init__()
        self.instance_mask = _InstanceMask(num_fields, field_emb_size, hidden_dim)
        self.input_proj = nn.Linear(input_dim, num_fields * field_emb_size) if input_dim != num_fields * field_emb_size else nn.Identity()
        self.fc = nn.Sequential(
            nn.Linear(num_fields * field_emb_size, hidden_dim),
            nn.ReLU(),
        )

    def forward(self, field_embs: list[Tensor], x: Tensor) -> Tensor:
        concat = self.input_proj(x)
        mask = self.instance_mask(field_embs)
        return self.fc(concat * mask)


class MaskNet(BaseModel):
    """MaskNet for CTR prediction (Serial version)."""

    def __init__(self, model_cfg: BaseModelConfig) -> None:
        super().__init__()

        self.embedding_fields: Mapping[str, FieldEntry] = model_cfg.embedding_fields
        self.field_names = list(self.embedding_fields.keys())

        # Embedding bags for all non-direct fields
        self.embedding_bags = nn.ModuleDict()
        for field_name, entry in self.embedding_fields.items():
            if entry.field_type == 0 and entry.concat_type == "direct":
                continue
            key = str(entry.field_index)
            if key not in self.embedding_bags:
                self.embedding_bags[key] = nn.EmbeddingBag(
                    num_embeddings=int(entry.dim),
                    embedding_dim=int(entry.emb_size),
                    mode="sum",
                )

        # Determine common embedding size (project if needed)
        self.field_names_no_direct = [n for n in self.field_names
                                      if not (self.embedding_fields[n].field_type == 0 and self.embedding_fields[n].concat_type == "direct")]
        self.num_fields = len(self.field_names_no_direct)

        # Use the first categorical field's emb_size as the common size
        if self.field_names_no_direct:
            self.field_emb_size = int(self.embedding_fields[self.field_names_no_direct[0]].emb_size)
        else:
            self.field_emb_size = 8

        # Project all field embeddings to a uniform size
        self.field_proj = nn.ModuleDict()
        for fn in self.field_names_no_direct:
            entry = self.embedding_fields[fn]
            if int(entry.emb_size) != self.field_emb_size:
                self.field_proj[fn] = nn.Linear(int(entry.emb_size), self.field_emb_size, bias=False)

        # MaskNet config
        mask_cfg: dict[str, Any] = model_cfg.mlp
        num_blocks = int(mask_cfg.get("num_blocks", 2))
        mask_hidden = list(mask_cfg.get("mask_hidden", [256]))
        reduction = int(mask_cfg.get("reduction_ratio", 4))

        # Direct-type continuous fields (not processed through mask blocks)
        self.direct_field_names = [n for n in self.field_names
                                   if self.embedding_fields[n].field_type == 0 and self.embedding_fields[n].concat_type == "direct"]
        direct_dim = sum(int(self.embedding_fields[n].dim) for n in self.direct_field_names)

        # Mask blocks (serial)
        concat_dim = self.num_fields * self.field_emb_size
        self.mask_blocks = nn.ModuleList()
        prev_dim = concat_dim
        for i in range(num_blocks):
            block_hidden = mask_hidden[i] if i < len(mask_hidden) else mask_hidden[-1]
            self.mask_blocks.append(_SerialMaskBlock(self.num_fields, self.field_emb_size, prev_dim, block_hidden))
            prev_dim = block_hidden

        # Final head (mask output + direct features)
        self.head = nn.Linear(prev_dim + direct_dim, 1)
        self._validate_fields(model_cfg)
        self.reset_parameters()

    def _validate_fields(self, model_cfg: BaseModelConfig) -> None:
        if not model_cfg.embedding_fields:
            raise ValueError("embedding_fields must be a non-empty mapping")

    def reset_parameters(self) -> None:
        for emb in self.embedding_bags.values():
            nn.init.xavier_uniform_(emb.weight)
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, feature_bags: Mapping[str, Mapping[str, Tensor]]) -> Tensor:
        first_offsets = feature_bags[self.field_names[0]]["offsets"]
        batch_size = int(first_offsets.size(0))
        device = next(self.parameters()).device

        # Embed each field (skip direct-type fields in mask blocks)
        field_embs: list[Tensor] = []
        for fn in self.field_names_no_direct:
            entry = self.embedding_fields[fn]
            emb = embed_one_field(
                self.embedding_bags[str(entry.field_index)],
                feature_bags[fn]["indices"],
                feature_bags[fn]["offsets"],
                feature_bags[fn]["weights"],
                device=device,
            )
            if fn in self.field_proj:
                emb = self.field_proj[fn](emb)
            field_embs.append(emb)

        if not field_embs:
            return torch.zeros(batch_size, device=device)

        # Serial mask blocks
        x = torch.cat(field_embs, dim=-1)
        for block in self.mask_blocks:
            x = block(field_embs, x)

        # Append direct-type features
        direct_feats: list[Tensor] = []
        for fn in self.direct_field_names:
            direct_feats.append(feature_bags[fn]["weights"].view(-1, int(self.embedding_fields[fn].dim)))
        if direct_feats:
            x = torch.cat([x] + direct_feats, dim=-1)

        logit = self.head(x).squeeze(-1)
        return torch.sigmoid(logit)
