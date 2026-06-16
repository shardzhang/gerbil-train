"""Classification losses.

Typical examples include:
    - binary cross-entropy
    - cross-entropy
    - sampled softmax
    - noise contrastive estimation (NCE)
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def nce_loss(
    hidden: Tensor,
    class_weight: Tensor,
    targets: Tensor,
    *,
    num_sampled: int = 100,
    noise_class_weight: Tensor | None = None,
) -> Tensor:
    """Noise Contrastive Estimation loss.

    Transforms multi-class classification into a binary task: distinguish true
    data from noise samples. Trains the model's own classification head without
    ever computing logits over all classes.

    :param hidden: Hidden state ``[batch_size, emb_dim]`` from the model.
    :param class_weight: Classification weight ``[num_classes, emb_dim]``.
    :param targets: True class indices ``[batch_size]``.
    :param num_sampled: Number of noise samples per batch element.
    :param noise_class_weight: Optional separate noise embedding.
        If ``None``, ``class_weight`` is reused for the noise model.
    :return: Scalar loss tensor.
    """
    batch_size = hidden.size(0)
    device = hidden.device
    num_classes = class_weight.size(0)

    # Sample noise classes uniformly
    noise = torch.randint(0, num_classes, (batch_size, num_sampled), device=device)

    # Positive score: [batch, 1]
    pos_w = class_weight[targets]
    pos_scores = (hidden * pos_w).sum(dim=-1, keepdim=True)

    # Noise scores: [batch, num_sampled]
    neg_w = class_weight[noise]
    neg_scores = torch.bmm(neg_w, hidden.unsqueeze(-1)).squeeze(-1)

    # Log-Q correction: log(K * P_noise) where P_noise = 1/num_classes
    log_k = torch.log(torch.tensor(float(num_sampled) / num_classes, device=device))

    # Stack positive and noise scores; positive label=1, noise labels=0
    all_scores = torch.cat([pos_scores, neg_scores], dim=-1)  # [batch, 1 + K]
    labels = torch.cat([
        torch.ones(batch_size, 1, device=device),
        torch.zeros(batch_size, num_sampled, device=device),
    ], dim=-1)

    return F.binary_cross_entropy_with_logits(all_scores - log_k, labels, reduction='mean')


def sampled_softmax_loss(
    hidden: Tensor,
    class_weight: Tensor,
    targets: Tensor,
    *,
    num_sampled: int = 100,
    class_bias: Tensor | None = None,
) -> Tensor:
    """Sampled softmax loss for large multi-class classification.

    Instead of computing logits over all ``num_classes`` classes, samples
    ``num_sampled`` negative classes per batch and computes softmax only over
    the positive class plus the sampled negatives.

    :param hidden: Hidden state ``[batch_size, emb_dim]`` from the model.
    :param class_weight: Classification weight ``[num_classes, emb_dim]``.
    :param targets: True class indices ``[batch_size]``.
    :param num_sampled: Number of negative samples per batch element.
    :param class_bias: Optional classification bias ``[num_classes]``.
    :return: Scalar loss tensor.
    """
    batch_size = hidden.size(0)
    device = hidden.device
    num_classes = class_weight.size(0)

    neg = torch.randint(0, num_classes, (batch_size, num_sampled), device=device)

    pos_w = class_weight[targets]
    pos_scores = (hidden * pos_w).sum(dim=-1, keepdim=True)
    if class_bias is not None:
        pos_scores = pos_scores + class_bias[targets].unsqueeze(-1)

    neg_w = class_weight[neg]
    neg_scores = torch.bmm(neg_w, hidden.unsqueeze(-1)).squeeze(-1)
    if class_bias is not None:
        neg_scores = neg_scores + class_bias[neg]

    all_scores = torch.cat([pos_scores, neg_scores], dim=-1)
    labels = torch.zeros(batch_size, dtype=torch.long, device=device)
    return F.cross_entropy(all_scores, labels)


__all__ = ["nce_loss", "sampled_softmax_loss"]
