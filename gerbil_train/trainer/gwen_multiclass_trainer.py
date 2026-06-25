"""Trainer for GwEN multi-class recommendation models.
prediction next movie which user could like.
"""

from __future__ import annotations

from typing import Any

import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from gerbil_train.config.train_config import TrainConfig
from gerbil_train.trainer.multi_trainer import MultiClassClassificationTrainer

__all__ = ["GwENMultiTrainer"]


class GwENMultiTrainer(MultiClassClassificationTrainer):
    """Trainer for GwEN multi-class recommendation models."""
    def __init__(self, model: nn.Module, train_cfg: TrainConfig, data_cfg: dict[str, Any] | None = None) -> None:
        super().__init__(model, train_cfg, data_cfg)
        self.model_name = "GwEN"

    def compute_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(logits, targets)
