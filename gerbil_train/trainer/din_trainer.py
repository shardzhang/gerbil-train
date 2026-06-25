"""Trainer for DIN (Deep Interest Network) binary classification models."""

from __future__ import annotations

from torch import nn

from gerbil_train.config.train_config import TrainConfig
from gerbil_train.trainer.binary_trainer import BinaryClassificationTrainer

__all__ = ["DINTrainer"]


class DINTrainer(BinaryClassificationTrainer):
    def __init__(self, model: nn.Module, config: TrainConfig, data_cfg: dict[str, str] | None = None) -> None:
        super().__init__(model, config, data_cfg)
        self.model_name = "DIN"
