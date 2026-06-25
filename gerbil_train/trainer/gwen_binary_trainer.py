"""Trainer for GwEN binary classification models."""

from __future__ import annotations

from torch import nn

from gerbil_train.config.train_config import TrainConfig
from gerbil_train.trainer.binary_trainer import BinaryClassificationTrainer

__all__ = ["GwENBinaryTrainer"]


class GwENBinaryTrainer(BinaryClassificationTrainer):
    def __init__(self, model: nn.Module, train_cfg: TrainConfig, data_cfg: dict[str, str]) -> None:
        super().__init__(model, train_cfg, data_cfg)
        self.model_name = "GwEN Binary"