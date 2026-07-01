"""Trainer for YouTubeDNN multi-class recommendation models."""

from __future__ import annotations

from typing import Any

import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from gerbil_train.config.train_config import TrainConfig
from gerbil_train.trainer.multi_trainer import MultiClassClassificationTrainer

__all__ = ["YouTubeDNNTrainer"]


class YouTubeDNNTrainer(MultiClassClassificationTrainer):
    """Trainer for YouTubeDNN model (multi-class classification)."""

    def __init__(self, model: nn.Module, train_cfg: TrainConfig, data_cfg: dict[str, Any] | None = None) -> None:
        super().__init__(model, train_cfg, data_cfg)
        self.model_name = "YouTubeDNN"
