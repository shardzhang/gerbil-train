"""Trainer for MV-DNN (pairwise ranking loss with negative sampling)."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from gerbil_train.config.train_config import TrainConfig
from gerbil_train.trainer.binary_trainer import BinaryClassificationTrainer

__all__ = ["MVDNNTrainer"]


class MVDNNTrainer(BinaryClassificationTrainer):
    def __init__(self, model: nn.Module, train_cfg: TrainConfig, data_cfg: dict[str, Any] | None = None) -> None:
        super().__init__(model, train_cfg, data_cfg)
        self.model_name = "MV-DNN"
        self.num_neg = int(train_cfg.loss.num_neg)
        self.item_vocab = int(model.fields_cfg[model.item_field].dim)

    def forward_step(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        feature_bags = batch["feature_bags"]
        model = self.model
        device = next(model.parameters()).device
        batch_size = list(feature_bags.values())[0]["offsets"].size(0)

        # Positive scores
        pos_scores = model(feature_bags)

        # Negative sampling + scores
        neg_scores_list = []
        for _ in range(self.num_neg):
            # Create negative item features
            neg_item_ids = torch.randint(0, self.item_vocab, (batch_size,), device=device)

            # Build negative feature dict: copy all user features, replace item
            neg_feat = dict(feature_bags)
            entry = model.fields_cfg[model.item_field]
            if entry.field_type == 0 and entry.concat_type == "direct":
                neg_feat[model.item_field] = {
                    "indices": torch.arange(batch_size * 2, device=device),
                    "offsets": torch.arange(batch_size, device=device),
                    "weights": neg_item_ids.float().view(-1, 1),
                }
            else:
                neg_feat[model.item_field] = {
                    "indices": neg_item_ids,
                    "offsets": torch.arange(batch_size, device=device),
                    "weights": torch.ones(batch_size, device=device),
                }
            # Also replace item side fields with negative samples
            for sf in model.item_side_fields:
                s_entry = model.fields_cfg[sf]
                neg_ids = torch.randint(0, int(s_entry.dim), (batch_size,), device=device)
                neg_feat[sf] = {
                    "indices": neg_ids,
                    "offsets": torch.arange(batch_size, device=device),
                    "weights": torch.ones(batch_size, device=device),
                }

            neg_scores = model(neg_feat)
            neg_scores_list.append(neg_scores)

        neg_scores = torch.stack(neg_scores_list, dim=1)

        # BPR loss
        pos_exp = pos_scores.unsqueeze(-1).expand_as(neg_scores)
        bpr_loss = -F.logsigmoid(pos_exp - neg_scores).mean()

        return {"loss": bpr_loss}

    def compute_loss(self, outputs: dict[str, torch.Tensor], batch: dict[str, Any]) -> torch.Tensor:
        return outputs["loss"]

    def compute_metrics(self, outputs: dict[str, torch.Tensor], batch: dict[str, Any]) -> dict[str, float]:
        return {"loss": float(outputs["loss"].item())}
