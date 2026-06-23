"""Shared embedding utilities for all models."""

from __future__ import annotations

import torch
from torch import Tensor, nn

__all__ = ["bag_to_padded", "embed_one_field", "to_device"]


def to_device(tensor: Tensor, device: torch.device) -> Tensor:
    return tensor if tensor.device == device else tensor.to(device)


def embed_one_field(emb: nn.EmbeddingBag, 
                    indices: Tensor, 
                    offsets: Tensor,
                    weights: Tensor, 
                    device: torch.device) -> Tensor:
    """Embeds a single field using the provided embedding bag.

    Args:
        emb (nn.EmbeddingBag): The embedding bag to use. 不需要解包，直接查表加和，一步完成 Embedding
        indices (Tensor): The indices of the elements to embed. shape: [total_num_indices]
        offsets (Tensor): The offsets for the embedding bag. shape: [batch_size]
        weights (Tensor): The weights for the embedding bag. shape: [total_num_indices]
        device (torch.device): The device to perform the computation on.

    Returns:
        Tensor: The embedded representation of the field. shape: [batch_size, embedding_dim]
    """
    indices = to_device(indices.long(), device)
    offsets = to_device(offsets.long(), device)
    weights = to_device(weights.float(), device)
    return emb(indices, offsets, per_sample_weights=weights)


def bag_to_padded(indices: Tensor, offsets: Tensor) -> tuple[Tensor, Tensor, int]:
    """Convert EmbeddingBag flat format to a padded sequence for attention/sequence models.

    EmbeddingBag stores variable-length sequences in a flat layout::
        indices  = [a0, a1, a2, b0, b1, c0, c1, c2, c3]   -- all tokens concatenated
        offsets  = [0, 3, 5]                                 -- start position of each sample

        sample A: indices[0:3]  = [a0, a1, a2]              -- 3 tokens
        sample B: indices[3:5]  = [b0, b1]                  -- 2 tokens
        sample C: indices[5:9]  = [c0, c1, c2, c3]          -- 4 tokens

    Attention / RNN / Transformer all require a dense ``[batch, seq_len]`` tensor.
    This function vectorizes the conversion via broadcast + gather + where.

    Step 1 — Compute ends and lengths::
        starts = [0, 3, 5]
        ends   = [3, 5, 9]             # offsets[1:] + [total]
        lengths = [3, 2, 4]            # min(end - start, max_seq_len)

    Step 2 — Build position matrix pos::
        pos = [[0, 1, 2, 3]]           # [1, max_seq_len], j = position in sequence

    Step 3 — gather_idx: each position's global index in indices::
        starts.unsqueeze(1) = [[0], [3], [5]]
        gather_idx = starts.unsqueeze(1) + pos
        gather_idx = [[0, 1, 2, 3],     # sample A: pull indices[0], indices[1], ...
                      [3, 4, 5, 6],     # sample B
                      [5, 6, 7, 8]]     # sample C
        clamp(max=total-1)

    Step 4 — mask: which positions are valid (within actual length)::
        mask = pos < lengths.unsqueeze(1)
        mask = [[T, T, T, F],           # sample A: first 3 valid
                [T, T, F, F],           # sample B: first 2 valid
                [T, T, T, T]]           # sample C: all valid

    Step 5 — torch.where: gather values for valid positions, fill 0 elsewhere::
        padded = torch.where(mask, indices[gather_idx], 0)
        padded = [[a0, a1, a2, 0],
                  [b0, b1, 0,  0],
                  [c0, c1, c2, c3]]
        lengths = [3, 2, 4]

    :param indices: ``[total_tokens]`` flat token IDs for all samples
    :param offsets: ``[batch]`` start position of each sample in ``indices``
    :return: ``(padded, lengths, max_seq_len)``
        padded:    ``[batch, max_seq_len]`` padded token ID matrix
        lengths:   ``[batch]`` actual sequence length per sample
        max_seq_len:  scalar, padded width
    """
    device = indices.device
    batch_size = offsets.size(0)
    total = indices.size(0)

    # [batch] end positions (append total as the end of the last sample)
    ends = torch.cat([offsets[1:], torch.tensor([total], device=device, dtype=indices.dtype)])
    # [batch] actual sequence lengths
    raw_lengths = ends - offsets
    max_seq_len = int(raw_lengths.max().item()) if batch_size > 0 else 0
    # [batch] lengths capped at max_seq_len
    lengths = torch.clamp(raw_lengths, max=max_seq_len)

    # [1, max_seq_len] position offsets
    pos = torch.arange(max_seq_len, device=device).unsqueeze(0)
    # [batch, max_seq_len] global index into ``indices``
    gather_idx = offsets.unsqueeze(1) + pos
    # avoid OOB
    gather_idx = torch.clamp(gather_idx, max=total - 1)

    # [batch, max_seq_len] valid position mask
    mask = pos < lengths.unsqueeze(1)
    # [batch, max_seq_len] gather values and zero-pad where mask is False
    # indices[gather_idx]: 花式索引, 输出形状 = gather_idx形状
    # weight_matrix[effective_index]: 布尔索引, 返回一维. 所有被 True 选中的元素被展平成一个一维列表，不保留原始形状
    padded = torch.where(mask, indices[gather_idx], torch.tensor(0, device=device, dtype=indices.dtype))
    return padded, lengths, max_seq_len
