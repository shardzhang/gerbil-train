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
    - ILS(intra-list similarity)： 针对单个用户，一般来说ILS值越大，单个用户推荐列表多样性越差

推荐算法的离线评价指标综述 - Tang AI的文章 - 知乎
https://zhuanlan.zhihu.com/p/584923052
"""

import torch

__all__ = ["ndcg_score"]

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


# 1. Precision（精确率）
# 推荐的结果里，有多少是用户真正喜欢的→ 越高说明推荐越准
# 2. Recall（召回率）
# 用户喜欢的东西，有多少被你推荐出来了→ 越高说明覆盖越全
# 3. F-measure（F1 分数）
# Precision + Recall 的综合分数→ 越高说明整体效果越好
# 4. Novelty（新颖度）
# 推荐结果的多样性 / 新奇度→ 越高说明推荐不重复、不单调

def presision(result_list, gt_list, top_k):
    """计算precision指标
    :param result_list: list of list, 每个list是一个用户的推荐结果
    :param gt_list: list of list, 每个list是一个用户的真实点击结果
    :param top_k: int, 推荐结果的长度
    :return: float, precision指标的值
    """
    count = 0.0
    for r, g in zip(result_list, gt_list):
        count += len(set(r).intersection(set(g)))
    return count / (top_k * len(result_list))

def recall(result_list, gt_list):
    """计算recall指标
    :param result_list: list of list, 每个list是一个用户的推荐结果
    :param gt_list: list of list, 每个list是一个用户的真实点击结果
    :return: float, recall指标的值
    """
    t = 0.0
    for r, g in zip(result_list, gt_list):
        t += 1.0 * len(set(r).intersection(set(g))) / len(g)
    return t / len(result_list)

def f_measure(result_list, gt_list, top_k, eps=1.0e-9):
    """计算f_measure指标
    :param result_list: list of list, 每个list是一个用户的推荐结果
    :param gt_list: list of list, 每个list是一个用户的真实点击结果
    :param top_k: int, 推荐结果的长度
    :param eps: float, 防止除零的极小值
    :return: float, f_measure指标的值
    """
    f = 0.0
    for r, g in zip(result_list, gt_list):
        recc = 1.0 * len(set(r).intersection(set(g))) / len(g)
        pres = 1.0 * len(set(r).intersection(set(g))) / top_k
        if recc + pres < eps:
            continue
        f += (2 * recc * pres) / (recc + pres)
    return f / len(result_list)

def novelty(result_list, s_u, top_k):
    """计算novelty指标
    :param result_list: list of list, 每个list是一个用户的推荐结果
    :param s_u: list of list, 每个list是一个用户的历史点击结果
    :param top_k: int, 推荐结果的长度
    :return: float, novelty指标的值
    """
    count = 0.0
    for r, g in zip(result_list, s_u):
        count += len(set(r) - set(g))
    return count / (top_k * len(result_list))

def hit_ratio(result_list, gt_list):
    """计算hit_ratio指标
    :param result_list: list of list, 每个list是一个用户的推荐结果
    :param gt_list: list of list, 每个list是一个用户的真实点击结果
    :return: float, hit_ratio指标的值
    """
    intersect_set = [len(set(r) & set(g)) for r, g in zip(result_list, gt_list)]
    return 1.0 * sum(intersect_set) / sum([len(gts) for gts in gt_list])

def NDCG(result_list, gt_list):
    """计算NDCG指标
    :param result_list: list of list, 每个list是一个用户的推荐结果
    :param gt_list: list of list, 每个list是一个用户的真实点击结果
    :return: float, NDCG指标的值
    """
    t = 0.0
    for re, gt in zip(result_list, gt_list):
        setgt = set(gt)
        indicator = np.asfarray([1 if r in setgt else 0 for r in re])
        sorted_indicator = np.ones(min(len(setgt), len(re)))
        if 1 in indicator:
            t+=np.sum(indicator / np.log2(1.0*np.arange(2,len(indicator)+ 2)))/\
               np.sum(sorted_indicator/np.log2(1.0*np.arange(2,len(sorted_indicator)+ 2)))
    return t/len(gt_list)

def MAP(result_list, gt_list, topk):
    """计算MAP指标
    :param result_list: list of list, 每个list是一个用户的推荐结果
    :param gt_list: list of list, 每个list是一个用户的真实点击结果
    :param topk: int, 推荐结果的长度
    :return: float, MAP指标的值
    """
    t = 0.0
    for re, gt in zip(result_list, gt_list):
        setgt = set(gt)
        indicator = np.asfarray([1 if r in setgt else 0 for r in re])
        t += np.mean([indicator[:i].sum(-1) / i for i in range(1, topk + 1)], axis=-1)
    return t/len(gt_list)