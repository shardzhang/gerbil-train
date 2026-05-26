"""Ranking metrics.

Currently implemented:
    - NDCG@k

Potential future additions:
    - MRR@k
    - Precision@k
    - Recall@k
    - MAP@k
    - AUC
    - F1@k
"""

import torch


def ndcg_score(y_true, y_score, k=5):
    """Compute NDCG@k for a single query.

    :param y_true: Tensor of shape (num_docs,) with true relevance scores
    :param y_score: Tensor of shape (num_docs,) with predicted relevance scores
    :param k: Rank position up to which NDCG is computed
    :return: NDCG@k score as a float
    """
    k = min(k, len(y_true))
    order = y_score.argsort(descending=True)[:k]
    ranked_y_true = y_true[order]

    gain = 2**ranked_y_true - 1
    discount = torch.log2(
        torch.arange(2, 2 + k, dtype=torch.float32, device=y_true.device)
    )
    dcg = (gain / discount).sum()

    ideal = torch.sort(y_true, descending=True).values[:k]
    ideal_gain = 2**ideal - 1
    idcg = (ideal_gain / discount).sum()

    res = dcg / idcg if idcg > 0 else 0.0
    return float(res)


__all__ = ["ndcg_score"]
