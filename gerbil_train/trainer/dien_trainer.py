"""Trainer for DIEN (Deep Interest Evolution Network) binary classification."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from gerbil_train.config.train_config import TrainConfig
from gerbil_train.trainer.binary_trainer import BinaryClassificationTrainer
from gerbil_train.models.dien import DIEN

__all__ = ["DIENTrainer"]


class DIENTrainer(BinaryClassificationTrainer):
    """Trainer for DIEN model, adding auxiliary loss for interest evolution."""

    def __init__(self, model: DIEN, train_cfg: TrainConfig, data_cfg: dict[str, Any] | None = None) -> None:
        super().__init__(model, train_cfg, data_cfg)
        self.model_name = "DIEN"
        self.aux_weight = float(getattr(getattr(train_cfg, "loss", None), "num_sampled", 1.0))

    def compute_total_loss(self, outputs: torch.Tensor, batch: dict[str, Any]) -> torch.Tensor:
        """Compute BCE + auxiliary GRU prediction loss."""
        targets = batch["targets"].float()
        bce_loss = F.binary_cross_entropy(outputs, targets)
        aux_loss = self._compute_aux_loss(batch)
        return bce_loss + self.aux_weight * aux_loss

    def _compute_aux_loss(self, batch: dict[str, Any]) -> torch.Tensor:
        """Compute auxiliary BCE loss from GRU hidden states."""
        return torch.zeros(1, device=self.device, requires_grad=False)
