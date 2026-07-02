"""Trainer for AFM (Attentional Factorization Machine) binary classification models."""

from __future__ import annotations

from typing import Any

from torch import nn

from gerbil_train.config.train_config import TrainConfig
from gerbil_train.trainer.binary_trainer import BinaryClassificationTrainer

__all__ = ["AFMTrainer"]


class AFMTrainer(BinaryClassificationTrainer):
    def __init__(self, model: nn.Module, train_cfg: TrainConfig, data_cfg: dict[str, Any] | None = None) -> None:
        super().__init__(model, train_cfg, data_cfg)
        self.model_name = "AFM"
