"""Trainer for DIN (Deep Interest Network) binary classification models."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from gerbil_train.config.train_config import TrainConfig
from gerbil_train.trainer.binary_trainer import BinaryClassificationTrainer

__all__ = ["DINTrainer"]


class DINTrainer(BinaryClassificationTrainer):
    def __init__(self, model: nn.Module, train_cfg: TrainConfig, data_cfg: dict[str, Any] | None = None) -> None:
        super().__init__(model, train_cfg, data_cfg)
        self.model_name = "DIN"


    def calculate_preference(self, batch_user, batch_items):
        """calculate the preference score of batch_user to batch_items with DINModel
        
        :param batch_user: (batch_size, f_num)
        :param batch_items: (batch_size, 1)
        :return: (batch_size, 1)
        """
        return self.model(batch_user, batch_items)


    @torch.no_grad()
    def predict(self, input, topk=10):
        """ predict the top-k items with the highest preference score for each user in test_instances
        
        :param test_instances: (batch_size, f_num)
        :param topk: the number of items to recommend for each user
        :return: (batch_size, topk) the recommended item IDs for each user
        """
        self.model.eval()

        input = input.to(self.device)
        batch_size = input.shape[0]
        
        # 1. 构建一个包含所有物品ID的张量
        # (item_num, )
        all_items = torch.arange(self.item_num, device=self.device)
        
        # 2. 计算每个用户对所有物品的偏好得分
        # 暴力全量排序 Brute-force Top-K. 推荐系统里最标准、最准确的评估方式
        scores = []
        for i in range(batch_size):
            # (1, f_num) -> (item_num, f_num)
            user_seq = input[i:i+1].repeat(all_items.shape[0], 1)
            
            # (item_num, ) -> (item_num, 1)
            items = all_items.view(-1, 1)
            
            # (item_num, 1) -> (item_num, )
            score = self.calculate_preference(user_seq, items).view(-1)
            scores.append(score)
        
        # (batch_size, item_num)
        scores = torch.stack(scores)
        
        # 3. 取Top-K
        topk_scores, topk_items = torch.topk(scores, k=topk, dim=-1)

        self.model.train()
        return topk_items