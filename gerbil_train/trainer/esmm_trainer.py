"""Trainer for ESMM (CTR + CVR multi-task) models.

Loss = BCE(pCTR, y_click) + BCE(pCTCVR, y_conversion)

where y_click ∈ {0,1} and y_conversion ∈ {0,1} are the click and conversion labels.
The targets tensor from the data pipeline has 2 columns: [click, conversion].
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from gerbil_train.config.train_config import TrainConfig
from gerbil_train.trainer.binary_trainer import BinaryClassificationTrainer

__all__ = ["ESMMTrainer"]


class ESMMTrainer(BinaryClassificationTrainer):
    def __init__(self, model: nn.Module, train_cfg: TrainConfig, data_cfg: dict[str, Any] | None = None) -> None:
        super().__init__(model, train_cfg, data_cfg)
        self.model_name = "ESMM"

    def forward_step(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        feature_bags = batch["feature_bags"]
        targets = batch["targets"]                          # [B, 2] = [click, conversion]
        outputs = self.model(feature_bags)

        y_click = targets[:, 0]
        y_conversion = targets[:, 1]

        loss_ctr = F.binary_cross_entropy(outputs["pctr"], y_click)
        loss_ctcvr = F.binary_cross_entropy(outputs["pctcvr"], y_conversion)
        total_loss = loss_ctr + loss_ctcvr

        return {"loss": total_loss, "loss_ctr": loss_ctr.item(), "loss_ctcvr": loss_ctcvr.item()}

    def compute_loss(self, outputs: dict[str, torch.Tensor], batch: dict[str, Any]) -> torch.Tensor:
        return outputs["loss"]

    def compute_metrics(self, outputs: dict[str, torch.Tensor], batch: dict[str, Any]) -> dict[str, float]:
        return {"loss": float(outputs["loss"].item())}
