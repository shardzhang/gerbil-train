"""Classification metrics.

Typical examples include:
    - Hit@K: whether the true class is in the top-K predictions
    - AUC: 随机抽一个正样本、一个负样本，模型把正样本排在前面的概率的期望值
    - GAUC: 对每个用户，随机抽一个正样本、一个负样本，模型把正样本排在前面的概率的期望值的期望值
    - AP: 平均精度
    - MAP: 平均平均精度，即对每个用户，取平均精度的期望值
    - accuracy
    - precision
    - recall
    - F1
    - log loss

These metrics are suitable for CTR prediction, binary scoring tasks, and
implicit-feedback classification evaluation.
"""

from __future__ import annotations

import torch

__all__ = ["auc", "average_precision", "gauc", "map_score", "hit_rate"]


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
    # AUC统计量公式
    # $$ \text{AUC} = \frac{\sum \text{pos_ranks} - \frac{P(P+1)}{2}}{P \times N} $$
    auc_value = (sum_pos_ranks - pos_count * (pos_count + 1) / 2) / (pos_count * neg_count)
    return float(auc_value.item())


def average_precision(labels: torch.Tensor, predictions: torch.Tensor) -> float:
    """Compute Average Precision (AP) for binary predictions.

    :param labels: Ground-truth labels ``[batch_size]``, each 0 or 1.
    :param predictions: Predicted probabilities ``[batch_size]`` in ``[0, 1]``.
    :return: AP score as a float.
    """
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
    total_weight = 0
    total_gauc = 0.0
    for uid in user_ids.unique():
        mask = user_ids == uid
        g_labels = labels[mask]
        g_scores = predictions[mask]
        n_pos = g_labels.sum()
        if 0 < n_pos < len(g_labels):   # 同时包含正负才参与
            w = int(mask.sum().item())
            total_weight += w
            total_gauc += w * auc(g_labels, g_scores)
    return total_gauc / max(total_weight, 1)


def map_score(
    user_ids: torch.Tensor,
    labels: torch.Tensor,
    predictions: torch.Tensor,
    weighted: bool = True,
) -> float:
    """Mean Average Precision (MAP), grouped by user_id.

    For each user (group), computes AP only if the group has both
    positive and negative samples.

    :param user_ids: ``[total_samples]`` user/query IDs
    :param labels: ``[total_samples]`` binary labels
    :param predictions: ``[total_samples]`` predicted scores
    :param weighted: ``True`` = weighted by group size, ``False`` = equal weight per group
    :return: MAP score as a float
    """
    total = 0.0
    total_weight = 0
    for uid in user_ids.unique():
        mask = user_ids == uid
        g_labels = labels[mask]
        g_scores = predictions[mask]
        n_pos = g_labels.sum()
        if 0 < n_pos < len(g_labels):
            ap = average_precision(g_labels, g_scores)
            w = int(mask.sum().item())
            total += ap * (w if weighted else 1)
            total_weight += w if weighted else 1
    return total / max(total_weight, 1)
