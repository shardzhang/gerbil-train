"""Trainer for PEPNet multi-task binary classification models."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from gerbil_train.config.train_config import TrainConfig
from gerbil_train.trainer.binary_trainer import BinaryClassificationTrainer

__all__ = ["PEPNetTrainer"]


class PEPNetTrainer(BinaryClassificationTrainer):
    def __init__(self, model: nn.Module, train_cfg: TrainConfig, data_cfg: dict[str, Any] | None = None) -> None:
        super().__init__(model, train_cfg, data_cfg)
        self.model_name = "PEPNet"

    def forward_step(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        targets = batch["targets"]
        task_outputs = self.model(batch["feature_bags"])

        num_tasks = len(task_outputs)
        if targets.dim() == 1:
            targets = targets.unsqueeze(-1).expand(-1, num_tasks)

        total_loss = torch.zeros(1, device=targets.device)
        for i, task_name in enumerate(task_outputs.keys()):
            total_loss = total_loss + F.binary_cross_entropy(task_outputs[task_name], targets[:, i])

        return {"loss": total_loss}

    def compute_loss(self, outputs: dict[str, torch.Tensor], batch: dict[str, Any]) -> torch.Tensor:
        return outputs["loss"]

    def compute_metrics(self, outputs: dict[str, torch.Tensor], batch: dict[str, Any]) -> dict[str, float]:
        return {"loss": float(outputs["loss"].item())}
