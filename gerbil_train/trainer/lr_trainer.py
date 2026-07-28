"""Trainer for FTRL (Follow The Regularized Leader) linear model."""

from __future__ import annotations

from torch import nn

from gerbil_train.config.train_config import FTRLOptimizerConfig, TrainConfig
from gerbil_train.trainer.binary_trainer import BinaryClassificationTrainer
from gerbil_train.optimizers.ftrl import FTRL

__all__ = ["FTRLTrainer"]


class FTRLTrainer(BinaryClassificationTrainer):
    """Trainer using FTRL-Proximal optimizer instead of Adam."""

    def __init__(self, model: nn.Module, train_cfg: TrainConfig, data_cfg: dict[str, Any] | None = None) -> None:
        super().__init__(model, train_cfg, data_cfg)
        self.model_name = "FTRL"

    def _create_optimizer(self, model: nn.Module, cfg: FTRLOptimizerConfig) -> FTRL:
        return FTRL(
            model.parameters(),
            alpha=float(cfg.lr),
            beta=float(cfg.beta),
            lambda1=float(cfg.lambda1),
            lambda2=float(cfg.lambda2),
        )
