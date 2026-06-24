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


def average_precision(labels: torch.Tensor, predictions: torch.Tensor) -> float:
    """Compute Average Precision (AP) for binary predictions.

    :param labels: Ground-truth labels ``[batch_size]``, each 0 or 1.
    :param predictions: Predicted probabilities ``[batch_size]`` in ``[0, 1]``.
    :return: AP score as a float.
    """
    labels = labels.flatten()
    predictions = predictions.flatten()
    sorted_indices = torch.argsort(predictions, descending=True)
    sorted_labels = labels[sorted_indices].float()
    pos_count = sorted_labels.sum()
    if pos_count == 0:
        return 0.0
    ranks = torch.arange(1, len(labels) + 1, device=labels.device).float()
    precisions = sorted_labels.cumsum(dim=0) / ranks
    return float((precisions * sorted_labels).sum() / pos_count)


def gauc(user_ids: torch.Tensor, labels: torch.Tensor, predictions: torch.Tensor) -> float:
    """Compute Group AUC (GAUC) weighted by group size.

    For each user (group), computes AUC only if the group has both
    positive and negative samples. The final GAUC is the weighted
    average across groups, weighted by the number of samples in each group.

    :param user_ids: ``[total_samples]`` user/query IDs
    :param labels: ``[total_samples]`` binary labels
    :param predictions: ``[total_samples]`` predicted scores
    :return: GAUC score as a float
    """
    labels = labels.flatten()
    predictions = predictions.flatten()
    user_ids = user_ids.flatten()
    total_weight = 0
    total_gauc = 0.0
    for uid in user_ids.unique():
        mask = user_ids == uid
        g_labels = labels[mask]
        g_scores = predictions[mask]
        if g_labels.float().sum() > 0 and (g_labels == 0).sum() > 0:
            w = int(mask.sum().item())
            total_weight += w
            total_gauc += w * auc(g_labels, g_scores)
    return total_gauc / max(total_weight, 1)


__all__ = ["auc", "average_precision", "gauc", "hit_rate"]
