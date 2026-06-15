"""Ranking losses

Paper: 
"""

import torch
import torch.nn as nn


LOSS_CHOICES = ("mse", "ranknet", "lambdarank", "listmle", "listnet")


def mse_loss(scores, labels):
    """pointwise
    :param scores: Tensor of shape (num_docs,) with predicted relevance scores
    :param labels: Tensor of shape (num_docs,) with true relevance scores
    :return: Scalar tensor representing the mean squared error loss
    """
    return nn.MSELoss()(scores, labels)


def ranknet_loss(s1, s2, y1, y2):
    """pairwise logistic loss for RankNET.

    RankNet does not learn an absolute score for each document directly.
    Instead, it learns pairwise preferences within the same query group:
    - for two documents ``doc_i`` and ``doc_j`` under the same query,
    - if ``doc_i`` is more relevant than ``doc_j`` in the ground truth,
    - the model should assign ``score_i > score_j``.

    :param s1: Tensor of shape (num_docs,) with predicted scores for the first document in each pair
    :param s2: Tensor of shape (num_docs,) with predicted scores for the second document in each pair
    :param y1: Tensor of shape (num_docs,) with true relevance scores for the first document in each pair
    :param y2: Tensor of shape (num_docs,) with true relevance scores for the second document in each pair
    :return: Scalar tensor representing the average pairwise loss
    """
    preferred_gap = torch.where(y1 > y2, s1 - s2, s2 - s1)
    # pairwise_loss = log(1 + exp(-preferred_gap)).
    # logaddexp(a, b) = log(exp(a) + exp(b))
    pairwise_loss = torch.logaddexp(-preferred_gap, torch.zeros_like(preferred_gap))
    loss = torch.where(y1 == y2, torch.zeros_like(pairwise_loss), pairwise_loss).mean()
    return loss


def ranknet_group_loss(scores, labels):
    """vectorized RankNet loss for one query group

      - Within a query, find all document pairs where the higher-label document
        should be ranked ahead of the lower-label document.
      - Compute ``log(1 + exp(s_j - s_i))`` for each valid pair.
      - Average the pairwise losses to obtain the query-level RankNet loss.

    :param scores: Tensor of shape (num_docs,) with predicted relevance scores
    :param labels: Tensor of shape (num_docs,) with true relevance scores
    :return: Scalar tensor representing the average pairwise loss
    """
    # valid pairs. label_i - label_j
    # Only optimize strictly ordered pairs; tied labels do not provide a preference signal.
    valid_pairs = labels[:, None] > labels[None, :]
    if not valid_pairs.any():
        return scores.sum() * 0.0

    # s_j - s_i
    pairwise_score_diff = scores[None, :] - scores[:, None]
    pairwise_loss = torch.logaddexp(
        pairwise_score_diff[valid_pairs],
        torch.zeros_like(pairwise_score_diff[valid_pairs]),
    )
    return pairwise_loss.mean()


def ranknet_group_loss_via_pairs(scores, labels):
    """RankNet loss for one query group computed via explicit pair tensors.

    This version keeps the same training target as ranknet_group_loss, but routes
    the actual pairwise logistic loss computation through ranknet_loss.

    :param scores: Tensor of shape (num_docs,) with predicted relevance scores
    :param labels: Tensor of shape (num_docs,) with true relevance scores
    :return: Scalar tensor representing the average pairwise loss
    """
    # valid pairs. label_i - label_j
    valid_pairs = labels[:, None] > labels[None, :]
    if not valid_pairs.any():
        return scores.sum() * 0.0

    pair_i, pair_j = torch.where(valid_pairs)
    s1 = scores[pair_i]
    s2 = scores[pair_j]
    y1 = labels[pair_i]
    y2 = labels[pair_j]
    return ranknet_loss(s1, s2, y1, y2)


def lambdarank_group_loss(scores, labels, k=None, eps=1e-10):
    """ΔNDCG-weighted RankNet loss / LambdaRank-style loss for a single query.

    :param scores: Predicted scores of shape (num_docs,).
    :param labels: Relevance labels of shape (num_docs,).
    :param k: (int, optional). If provided, optimize NDCG@k. If None, optimize full-list NDCG.
    :param eps: (float, optional). Numerical stability constant.
    :return: tuple[torch.Tensor, bool]:
        - scalar loss
        - whether this query group is valid
    """
    device = scores.device
    labels = labels.to(scores.dtype)
    n = len(labels)
    if n <= 1:
        return scores.sum() * 0.0, False

    # 1. Sort documents by predicted score and derive their predicted ranks.
    # Forward lookup: predicted rank -> document index.
    rank2doc_map = torch.argsort(scores, descending=True)
    # Reverse lookup: document index -> predicted rank.
    doc2rank_map = torch.empty_like(rank2doc_map)
    doc2rank_map[rank2doc_map] = torch.arange(n, device=device)

    # 2. Valid preference pairs satisfy label_i > label_j.
    pair_mask = labels[:, None] > labels[None, :]
    if not pair_mask.any():
        return scores.sum() * 0.0, False
    i, j = pair_mask.nonzero(as_tuple=True)

    # If optimizing NDCG@k, keep only pairs that can affect the top-k ranking.
    if k is not None:
        topk_mask = (doc2rank_map[i] < k) | (doc2rank_map[j] < k)
        if not topk_mask.any():
            return scores.sum() * 0.0, False
        i = i[topk_mask]
        j = j[topk_mask]

    # 3. Compute |ΔDCG| under the current predicted ordering.
    gains = torch.pow(2.0, labels) - 1.0
    ranks = doc2rank_map.to(scores.dtype)
    discounts = 1.0 / torch.log2(ranks + 2.0)
    if k is not None:
        discounts = discounts * (doc2rank_map < k).to(scores.dtype)
    delta_dcg = torch.abs((gains[i] - gains[j]) * (discounts[i] - discounts[j]))

    # 4. Compute IDCG@k.
    ideal_labels, _ = torch.sort(labels, descending=True)
    ideal_len = n if k is None else min(k, n)
    ideal_labels = ideal_labels[:ideal_len]
    ideal_gains = torch.pow(2.0, ideal_labels) - 1.0
    ideal_discounts = 1.0 / torch.log2(
        torch.arange(ideal_len, device=device, dtype=scores.dtype) + 2.0
    )
    idcg = torch.sum(ideal_gains * ideal_discounts).clamp_min(eps)

    # 5. Compute |ΔNDCG| = |ΔDCG| / IDCG.
    delta_ndcg = delta_dcg / idcg

    # 6. Compute the RankNet pairwise logistic loss.
    diff = scores[j] - scores[i]
    pairwise_loss = torch.nn.functional.softplus(diff)

    # 7. Weight the RankNet loss by ΔNDCG.
    # loss = |ΔNDCG| × softplus(diff)
    lambdarank_loss = delta_ndcg * pairwise_loss

    # 8. The LambdaRank lambda term is |ΔNDCG| × sigmoid(diff).
    # lambda = delta_ndcg * torch.sigmoid(diff)
    return lambdarank_loss.mean(), True


def lambdarank_batch_loss(scores, labels, query_group_ids, k=None, eps=1e-10):
    """Compute LambdaRank-style batch loss by grouping documents by query id.

    This function computes a ΔNDCG-weighted pairwise RankNet loss separately
    for each query group, then returns the mean loss over all valid query groups.

    A query group is considered valid if it contains at least one effective pair
    satisfying: label_i > label_j

    Query groups with no valid pairwise preference signal, such as groups where
    all labels are identical, are excluded from the final average.

    :param scores: Predicted relevance scores of shape (batch_size,).
    :param labels: Ground-truth relevance labels of shape (batch_size,).
    :param query_group_ids: Query/group identifiers of shape (batch_size,). Documents with the same query id are treated as belonging to the same ranking group, and pairwise loss is computed only within each group.
    :param k: (int, optional) If provided, compute the underlying group loss using NDCG@k. If None, use the full ranking list.
    :param eps: (float, optional) Small constant for numerical stability, typically used to avoid division by zero when IDCG is zero. Default is 1e-10.
    :return: A scalar tensor representing the mean LambdaRank loss over all valid query groups in the batch.
    """
    query_group_ids = query_group_ids.to(scores.device)
    unique_qids = torch.unique(query_group_ids)
    group_losses = []
    for qid in unique_qids:
        mask = query_group_ids == qid
        group_scores = scores[mask]
        group_labels = labels[mask]
        group_loss, is_valid = lambdarank_group_loss(
            group_scores, group_labels, k=k, eps=eps
        )
        if is_valid:
            group_losses.append(group_loss)

    if not group_losses:
        return scores.sum() * 0.0
    return torch.stack(group_losses).mean()


def listmle_loss(scores, labels):
    """ListMLE loss based on the Plackett-Luce model.

    :param scores: Tensor of shape (num_docs,) with predicted relevance scores
    :param labels: Tensor of shape (num_docs,) with true relevance scores
    :return: Scalar tensor representing the listwise loss
    """
    sorted_idx = torch.argsort(labels, descending=True)
    scores_sorted = scores[sorted_idx]
    cumsum = torch.logcumsumexp(torch.flip(scores_sorted, dims=[0]), dim=0)
    cumsum = torch.flip(cumsum, dims=[0])
    loss = -(scores_sorted - cumsum).mean()
    return loss


def listnet_loss(scores, labels):
    """ListNet top-1 listwise loss

    loss = -sum_i P_i * log(Q_i)
      - ``P_i = exp(label_i) / sum_j exp(label_j)`` is the probability
        distribution induced by the ground-truth labels.
      - ``Q_i = exp(score_i) / sum_j exp(score_j)`` is the probability
        distribution induced by the predicted scores.

    :param scores: Tensor of shape (num_docs,) with predicted relevance scores
    :param labels: Tensor of shape (num_docs,) with true relevance scores
    :return: Scalar tensor representing the listwise loss
    """
    labels = labels.to(scores.dtype)
    label_probs = torch.softmax(labels, dim=0)
    log_score_probs = torch.log_softmax(scores, dim=0)
    return -torch.sum(label_probs * log_score_probs)


def compute_loss(loss_name, scores, labels, k=None):
    """Dispatch ranking loss by name.

    :param loss_name: Ranking loss name
    :param scores: Predicted scores for one query group
    :param labels: Ground-truth relevance labels for one query group
    :param k: Optional ranking cutoff used by LambdaRank-style losses
    :return: Scalar loss tensor
    """
    if loss_name == "mse":
        return mse_loss(scores, labels)
    if loss_name == "ranknet":
        return ranknet_group_loss_via_pairs(scores, labels)
    if loss_name == "lambdarank":
        loss, _ = lambdarank_group_loss(scores, labels, k=k)
        return loss
    if loss_name == "listmle":
        return listmle_loss(scores, labels)
    if loss_name == "listnet":
        return listnet_loss(scores, labels)
    raise ValueError(f"Unknown loss: {loss_name}")


__all__ = [
    "LOSS_CHOICES",
    "compute_loss",
    "lambdarank_batch_loss",
    "lambdarank_group_loss",
    "listmle_loss",
    "listnet_loss",
    "mse_loss",
    "ranknet_group_loss",
    "ranknet_group_loss_via_pairs",
    "ranknet_loss",
]
