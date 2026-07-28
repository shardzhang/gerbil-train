"""SSL (Self-Supervised Learning) for sequential recommendation.

Uses contrastive learning on behavior sequences via data augmentation:
  - Random mask: randomly mask items in the sequence
  - Random crop: take a random contiguous subsequence

Two augmented views of the same sequence are pulled together (positive pairs)
while views from different sequences are pushed apart (InfoNCE loss).

Reference: https://arxiv.org/abs/2104.06879 (CL4SRec)
"""

from __future__ import annotations

from typing import Mapping, Any

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from gerbil_train.config.model_config import BaseModelConfig, FieldEntry
from gerbil_train.models.base_model import BaseModel

__all__ = ["SSLModel"]


class SSLModel(BaseModel):
    """Self-supervised contrastive learning model for item sequences."""

    def __init__(self, model_cfg: BaseModelConfig) -> None:
        super().__init__()
        self._validate_fields(model_cfg)

        self.fields_cfg: Mapping[str, FieldEntry] = model_cfg.embedding_fields

        ssl_cfg: dict[str, Any] = model_cfg.mlp
        self.item_field = str(ssl_cfg["item_field"])
        self.emb_size = int(self.fields_cfg[self.item_field].emb_size)
        self.vocab_size = int(self.fields_cfg[self.item_field].dim)
        self.temperature = float(ssl_cfg.get("temperature", 0.5))
        self.mask_ratio = float(ssl_cfg.get("mask_ratio", 0.15))

        # Item embedding
        self.item_embedding = nn.Embedding(self.vocab_size + 1, self.emb_size, padding_idx=self.vocab_size)

        # Sequence encoder (mean pooling + linear projection)
        self.encoder_proj = nn.Linear(self.emb_size, self.emb_size)

        # Projection head for contrastive learning
        self.projection = nn.Sequential(
            nn.Linear(self.emb_size, self.emb_size),
            nn.ReLU(),
            nn.Linear(self.emb_size, self.emb_size),
        )

        self.reset_parameters()

    def _validate_fields(self, model_cfg: BaseModelConfig) -> None:
        if not model_cfg.embedding_fields:
            raise ValueError("embedding_fields must be a non-empty mapping")
        if "item_field" not in model_cfg.mlp:
            raise ValueError("SSL config must specify item_field")

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.item_embedding.weight)
        nn.init.xavier_uniform_(self.encoder_proj.weight)
        nn.init.zeros_(self.encoder_proj.bias)

    def encode(self, item_ids: Tensor, lengths: Tensor) -> Tensor:
        """Encode a batch of sequences into embeddings.

        :param item_ids: [B, T] padded item IDs
        :param lengths: [B] actual sequence lengths
        :return: [B, d] sequence embeddings
        """
        emb = self.item_embedding(item_ids)                                # [B, T, d]
        mask = torch.arange(item_ids.size(1), device=item_ids.device).unsqueeze(0) < lengths.unsqueeze(1)
        emb = emb * mask.unsqueeze(-1)
        seq_emb = emb.sum(dim=1) / lengths.unsqueeze(-1).clamp(min=1).float()
        return self.encoder_proj(seq_emb)

    def _random_mask(self, item_ids: Tensor, lengths: Tensor) -> Tensor:
        """Randomly mask items in the sequence (excluding padding)."""
        masked = item_ids.clone()
        B, T = masked.shape
        mask_token = self.vocab_size
        for s in range(B):
            L = int(lengths[s].item())
            if L <= 1:
                continue
            num_mask = max(1, int(L * self.mask_ratio))
            indices = torch.randperm(L, device=masked.device)[:num_mask]
            masked[s, indices] = mask_token
        return masked

    def _random_crop(self, item_ids: Tensor, lengths: Tensor) -> tuple[Tensor, Tensor]:
        """Randomly crop a contiguous subsequence."""
        cropped = item_ids.clone()
        new_lengths = lengths.clone()
        B, T = cropped.shape
        for s in range(B):
            L = int(lengths[s].item())
            if L <= 1:
                continue
            crop_len = max(1, int(L * 0.8))
            start = torch.randint(0, L - crop_len + 1, (1,), device=cropped.device).item()
            cropped_slice = cropped[s, start:start + crop_len].clone()
            cropped[s, :crop_len] = cropped_slice
            cropped[s, crop_len:] = self.vocab_size
            new_lengths[s] = crop_len
        return cropped, new_lengths

    def forward(self, item_ids: Tensor, lengths: Tensor) -> dict[str, Tensor]:
        """Generate two augmented views and their embeddings.

        :param item_ids: [B, T] padded item IDs
        :param lengths: [B] sequence lengths
        :return: dict with view embeddings
        """
        mask_id = self.vocab_size

        # View 1: random mask
        masked_ids = self._random_mask(item_ids, lengths)
        h1 = self.encode(masked_ids, lengths)

        # View 2: random crop
        cropped_ids, crop_lengths = self._random_crop(item_ids, lengths)
        h2 = self.encode(cropped_ids, crop_lengths)

        return {"view1": h1, "view2": h2}

    def compute_contrastive_loss(self, h1: Tensor, h2: Tensor) -> Tensor:
        """InfoNCE loss between two augmented views.

        :param h1: [B, d] embeddings from view 1
        :param h2: [B, d] embeddings from view 2
        :return: scalar loss
        """
        z1 = F.normalize(self.projection(h1), dim=-1)
        z2 = F.normalize(self.projection(h2), dim=-1)

        # Cosine similarity matrix: [B, B]
        sim = torch.mm(z1, z2.T) / self.temperature

        # Positive pairs: diagonal
        labels = torch.arange(sim.size(0), device=sim.device)
        loss = F.cross_entropy(sim, labels) + F.cross_entropy(sim.T, labels)
        return loss / 2

    def embed(self, item_ids: Tensor, lengths: Tensor) -> Tensor:
        """Get sequence embedding (for inference)."""
        return self.encode(item_ids, lengths)
