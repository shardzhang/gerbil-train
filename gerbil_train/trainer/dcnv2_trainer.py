"""Trainer for DCNv2 (Deep & Cross Network V2) binary classification models."""

from __future__ import annotations

from typing import Any

from torch import nn

from gerbil_train.config.train_config import TrainConfig
from gerbil_train.trainer.binary_trainer import BinaryClassificationTrainer

__all__ = ["DCNv2Trainer"]


class DCNv2Trainer(BinaryClassificationTrainer):
    def __init__(self, model: nn.Module, train_cfg: TrainConfig, data_cfg: dict[str, Any] | None = None) -> None:
        super().__init__(model, train_cfg, data_cfg)
        self.model_name = "DCNv2"
