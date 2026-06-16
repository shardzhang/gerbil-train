"""Classification metrics.

Typical examples include:
    - AUC
    - log loss
    - accuracy
    - precision
    - recall
    - F1
    - Hit@K: whether the true class is in the top-K predictions

These metrics are suitable for CTR prediction, binary scoring tasks, and
implicit-feedback classification evaluation.
"""

from __future__ import annotations

import torch


def hit_rate(
    logits: torch.Tensor,
    targets: torch.Tensor,
    k: int,
) -> float:
    """Compute Hit@K: whether the true class is in the top-K predictions.

    :param logits: Predicted logits ``[batch_size, num_classes]``.
    :param targets: True class indices ``[batch_size]``.
    :param k: Number of top predictions to consider.
    :return: Hit@K as a float in ``[0, 1]``.
    """
    if logits.size(1) <= 1:
        return 1.0
    k = min(k, int(logits.size(1)))
    topk_indices = torch.topk(logits, k=k, dim=1).indices
    hits = (topk_indices == targets.unsqueeze(1)).any(dim=1).float()
    return float(hits.mean().item())


def auc(labels: torch.Tensor, predictions: torch.Tensor) -> float:
    """Compute AUC (Area Under the ROC Curve) for binary predictions.

    :param labels: Ground-truth labels ``[batch_size]``, each 0 or 1.
    :param predictions: Predicted probabilities ``[batch_size]`` in ``[0, 1]``.
    :return: AUC score as a float.
    """
    labels = labels.flatten()
    predictions = predictions.flatten()
    sorted_indices = torch.argsort(predictions, descending=True)
    sorted_labels = labels[sorted_indices].float()
    n = len(sorted_labels)
    pos_count = sorted_labels.sum()
    neg_count = n - pos_count
    if pos_count == 0 or neg_count == 0:
        return 0.5
    # Rank from n (highest) down to 1 (lowest)
    ranks = torch.arange(n, 0, -1, device=labels.device).float()
    sum_pos_ranks = (sorted_labels * ranks).sum()
    auc_value = (sum_pos_ranks - pos_count * (pos_count + 1) / 2) / (pos_count * neg_count)
    return float(auc_value.item())


__all__ = ["auc", "hit_rate"]
