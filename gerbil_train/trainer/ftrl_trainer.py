"""Trainer for FTRL (Follow The Regularized Leader) linear model."""

from __future__ import annotations

from typing import Any

from torch import nn

from gerbil_train.config.train_config import TrainConfig
from gerbil_train.trainer.binary_trainer import BinaryClassificationTrainer
from gerbil_train.optimizers.ftrl import FTRL

__all__ = ["FTRLTrainer"]


class FTRLTrainer(BinaryClassificationTrainer):
    """Trainer using FTRL-Proximal optimizer instead of Adam."""

    def __init__(self, model: nn.Module, train_cfg: TrainConfig, data_cfg: dict[str, Any] | None = None) -> None:
        super().__init__(model, train_cfg, data_cfg)
        self.model_name = "FTRL"

    def _create_optimizer(self, model: nn.Module, cfg: Any) -> FTRL:
        return FTRL(
            model.parameters(),
            alpha=float(getattr(cfg, "lr", 0.1)),
            beta=float(getattr(cfg, "beta", 1.0)),
            lambda1=float(getattr(cfg, "lambda1", 1.0)),
            lambda2=float(getattr(cfg, "lambda2", 1.0)),
        )
