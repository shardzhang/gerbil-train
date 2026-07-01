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

__all__ = ["nce_loss", "sampled_softmax_loss"]


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


def uniform_sampled_softmax(self, batch_users: torch.Tensor, batch_labels: torch.Tensor, N: int) -> torch.Tensor:
    """sampled softmax loss
    TODO: 是否可以直接调用tf.nn.sampled_softmax_loss来计算损失?

    :param: batch_users: (batch_size, f_num)
    :param: batch_labels: (batch_size, 1)
    :param: N: negative sample number
    :return: loss. scalar
    """
    batch_size = batch_users.shape[0]        

    # generate samples, the first column is positive labels, and the rest are negative labels
    # (batch_size, N + 1)
    labels = torch.full((batch_size, N + 1), -1, device=self.device, dtype=torch.int64)
    labels[:, 0:1] = batch_labels      # positive labels
    labels[:, 1:] = torch.randint(     # random negative labels
        low=0, 
        high=self.item_num, 
        size=(batch_size, N), 
        device=self.device,
    )

    # the effective index of samples, which is the index of items that are not filled with the absence (item_num)
    # (batch_size, N + 1)
    effective_index = torch.full(labels.shape, True, device=self.device, dtype=torch.bool)
    # the first column of samples is positive labels, so the effective index of the first column is always True
    # the effective index of the rest columns is True if the negative label is not equal to the positive label, otherwise False
    effective_index[:, 1:] = labels[:, 0:1] != labels[:, 1:]

    # compute the log of sampling probability, which is used to correct the bias caused by negative sampling
    # (batch_size, N + 1)
    log_q_matrix = torch.full(labels.shape, 0.0, device=self.device, dtype=torch.float32)

    # 负采样概率: q = 每个用户的有效负样本数 / (总物品数 - 1)
    # 每个用户的有效的所有负样本共享同一个log(q)
    log_q_matrix[:, 1:][effective_index[:, 1:]] = torch.log(
        effective_index[:, 1:].sum(-1).view(batch_size, 1) * 1.0 / (self.item_num - 1)
    ).expand(batch_size, N)[effective_index[:, 1:]]

    # (batch_size, N + 1)
    user_index = torch.arange(batch_size, device=self.device).view(-1, 1).expand(labels.shape)
    # (batch_size * effective_index_len, )
    # 用和原张量同形状的布尔张量做索引，会把原张量中所有True位置的元素展平成一维张量
    user_index = user_index[effective_index]

    # (batch_size * effective_index_len, )
    labels = labels[effective_index]
    # (batch_size * effective_index_len, 1)
    labels = labels.view(-1, 1)

    # (batch_size * effective_index_len, f_num)
    # 把每个用户的特征重复取出effective_index_len次, 拼成一个更长的矩阵, 和samples中的标签一一对应
    batch_users = batch_users[user_index]

    # (batch_size, N + 1)
    # the preference score of the positive and negative samples, which is computed by DINModel, and the bias caused by negative sampling is corrected with log_q_matrix
    o_pi = torch.full(log_q_matrix.shape, -1.0e9, device=self.device, dtype=torch.float32)

    # (batch_size * effective_index_len, )
    logits = self.DINModel(batch_user=batch_users, batch_label=labels)[:, 0]
    # 布尔掩码索引赋值: 只操作两个张量中掩码为True的位置
    o_pi[effective_index] = logits - log_q_matrix[effective_index]

    # compute the sampled softmax loss with log-sum-exp trick for numerical stability
    # (batch_size, ) -> scalar
    return (-o_pi[:, 0] + torch.logsumexp(o_pi, dim=1)).mean(-1)
