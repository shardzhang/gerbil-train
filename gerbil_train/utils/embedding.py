"""Shared embedding utilities for all models."""

from __future__ import annotations

import torch
from torch import Tensor, nn

__all__ = ["embed_one_field", "to_device"]


def to_device(tensor: Tensor, device: torch.device) -> Tensor:
    return tensor if tensor.device == device else tensor.to(device)


def embed_one_field(
    emb: nn.EmbeddingBag, indices: Tensor, offsets: Tensor,
    weights: Tensor, device: torch.device,
) -> Tensor:
    indices = to_device(indices.long(), device)
    offsets = to_device(offsets.long(), device)
    weights = to_device(weights.float(), device)
    return emb(indices, offsets, per_sample_weights=weights)
