"""Trainer for BPR (Bayesian Personalized Ranking) models.

Uses pairwise BPR loss with negative sampling during training.
During validation/evaluation, uses AUC (pointwise) since we have
implicit feedback with positive/negative labels.
"""

from __future__ import annotations

from typing import Any, Mapping

import torch
from torch import nn
import torch.nn.functional as F

from gerbil_train.config.train_config import TrainConfig
from gerbil_train.trainer.binary_trainer import BinaryClassificationTrainer

__all__ = ["BPRTrainer"]


class BPRTrainer(BinaryClassificationTrainer):
    def __init__(self, model: nn.Module, train_cfg: TrainConfig, data_cfg: dict[str, Any] | None = None) -> None:
        super().__init__(model, train_cfg, data_cfg)

        self.model_name = "BPR"
        self.model = model
        self.num_neg = int(train_cfg.optimizer.bpr_num_neg)
        self.item_vocab = int(model.fields_cfg[model.item_field].dim)


    def forward_step(self, batch: dict[str, Any]) -> torch.Tensor:
        feature_bags: Mapping[str, Mapping[str, torch.Tensor]] = batch["feature_bags"]
        targets = batch["targets"]
        user_field: str = self.model.user_field
        item_field: str = self.model.item_field

        # Positive scores
        pos_scores = self.model(feature_bags)

        batch_size = targets.size(0)
        device = pos_scores.device
        
        # Negative sampling + scores
        neg_scores_list = []
        for _ in range(self.num_neg):
            neg_item_ids = torch.randint(0, self.item_vocab, (batch_size,), device=device)
            neg_feature_bags = {
                user_field: {
                    "indices": feature_bags[user_field]["indices"],
                    "offsets": feature_bags[user_field]["offsets"],
                    "weights": feature_bags[user_field]["weights"],
                },
                item_field: {
                    "indices": neg_item_ids,
                    "offsets": torch.arange(batch_size, device=device),
                    "weights": torch.ones(batch_size, device=device),
                },
            }
            neg_scores = self.model(neg_feature_bags)
            neg_scores_list.append(neg_scores)

        # [batch_size, num_neg]
        neg_scores = torch.stack(neg_scores_list, dim=1)

        # BPR loss: -mean(log(sigmoid(pos - neg)))
        # pos_scores: [B], neg_scores: [B, num_neg]
        pos_exp = pos_scores.unsqueeze(-1).expand_as(neg_scores)
        bpr_loss = -F.logsigmoid(pos_exp - neg_scores).mean()
        return {"loss": bpr_loss, "pos_scores": pos_scores, "neg_scores": neg_scores}


    def compute_loss(self, outputs: dict[str, torch.Tensor], batch: dict[str, Any]) -> torch.Tensor:
        return outputs["loss"]


    def compute_metrics(self, outputs: dict[str, torch.Tensor], batch: dict[str, Any]) -> dict[str, float]:
        return {"loss": float(outputs["loss"].item())}
