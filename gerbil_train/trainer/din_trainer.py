"""Trainer for DIN (Deep Interest Network) binary classification models."""

# /Users/dazhang/PycharmProject/RecForest/lib/DIN_trainer.py

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from gerbil_train.config.train_config import GwENTrainConfig
from gerbil_train.trainer.base_trainer import BaseTrainer
from gerbil_train.utils import BatchInspector

__all__ = ["DINTrainer", "DINTrainingResult"]


@dataclass
class DINTrainingResult:
    train_loss_history: list[float]
    val_auc_history: list[float]
    best_auc: float


class DINTrainer(BaseTrainer):
    def __init__(self, model: nn.Module, config: GwENTrainConfig) -> None:
        
        optimizer_cfg = config.optimizer
        scheduler_cfg = config.scheduler
        checkpoint_cfg = config.checkpoint
        early_stop_cfg = config.early_stop
        logging_cfg = config.logging

        optimizer = optim.Adam(
            model.parameters(), 
            lr=float(optimizer_cfg.lr or 1e-3), 
            weight_decay=float(optimizer_cfg.weight_decay or 0.0)
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, 
                mode=str(scheduler_cfg.mode),
                factor=float(scheduler_cfg.factor), 
                patience=int(scheduler_cfg.patience),
            ) if scheduler_cfg.enabled else None
        
        device = config.device or ("cuda" if torch.cuda.is_available() else "cpu")
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"

        super().__init__(
            model=model, 
            optimizer=optimizer, 
            scheduler=scheduler, 
            device=device,
            gradient_clip_norm=None,
            monitor=str(checkpoint_cfg.monitor or "val_auc"),
            monitor_mode=str(checkpoint_cfg.mode or "max"),
            patience=0 if not early_stop_cfg.enabled else int(early_stop_cfg.patience),
            best_checkpoint_path=checkpoint_cfg.path,
            best_metric=None, 
            wait=0, 
            seed=config.seed,
        )
        self.config = config
        self.epochs = int(config.epochs)
        self.train_loader: DataLoader | None = None
        self.validation_loader: DataLoader | None = None
        self.test_loader: DataLoader | None = None
        self.model_name = "DIN"
        self.metric_name = "AUC"
        self.val_metric_history: list[float] = []
        self.plot_path = Path(logging_cfg.plot_path) if logging_cfg.plot_path is not None else None

        if config.inspector.enabled:
            self.set_batch_inspector(BatchInspector(
                log_first=config.inspector.log_first,
                log_every=config.inspector.log_every,
            ))

    def build_result(self) -> DINTrainingResult:
        return DINTrainingResult(
            train_loss_history=list(self.train_loss_history),
            val_auc_history=list(self.val_metric_history),
            best_auc=self.best_metric or 0.0,
        )
    
    def compute_loss(self, outputs: torch.Tensor, batch: Any) -> torch.Tensor:
        import torch.nn.functional as F
        return F.binary_cross_entropy(outputs, batch["targets"].float())
