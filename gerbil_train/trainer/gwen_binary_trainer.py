"""Trainer for GwEN binary classification models."""

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

__all__ = ["GwENBinaryTrainer", "GwENBinaryTrainingResult"]


@dataclass
class GwENBinaryTrainingResult:
    train_loss_history: list[float]
    val_auc_history: list[float]
    best_auc: float


class GwENBinaryTrainer(BaseTrainer):
    def __init__(self, model: nn.Module, train_cfg: GwENTrainConfig, data_cfg: dict[str, Any]) -> None:
        optimizer_cfg = train_cfg.optimizer
        scheduler_cfg = train_cfg.scheduler
        checkpoint_cfg = train_cfg.checkpoint
        early_stop_cfg = train_cfg.early_stop
        logging_cfg = train_cfg.logging

        optimizer = optim.Adam(
            model.parameters(),
            lr=float(optimizer_cfg.lr or 1e-3),
            weight_decay=float(optimizer_cfg.weight_decay or 0.0),
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode=str(scheduler_cfg.mode),
                factor=float(scheduler_cfg.factor), patience=int(scheduler_cfg.patience),
            ) if scheduler_cfg.enabled else None
        
        device = train_cfg.device or ("cuda" if torch.cuda.is_available() else "cpu")
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"

        super().__init__(
            model=model, 
            optimizer=optimizer, 
            scheduler=scheduler, device=device,
            gradient_clip_norm=None,
            monitor=str(checkpoint_cfg.monitor or "val_auc"),
            monitor_mode=str(checkpoint_cfg.mode or "max"),
            patience=0 if not early_stop_cfg.enabled else int(early_stop_cfg.patience),
            best_checkpoint_path=checkpoint_cfg.path,
            best_metric=None, 
            wait=0, 
            seed=train_cfg.seed,
        )

        self.config = train_cfg
        self.epochs = int(train_cfg.epochs)
        self.train_loader: DataLoader | None = None
        self.validation_loader: DataLoader | None = None
        self.test_loader: DataLoader | None = None
        self.model_name = "GwEN Binary"
        self.metric_name = "AUC"
        self.val_metric_history: list[float] = []
        self.plot_path = Path(logging_cfg.plot_path) if logging_cfg.plot_path is not None else None

        if data_cfg is not None:
            self.setup_total_train_samples(data_cfg, train_cfg.data.batch_size)
        
        if train_cfg.inspector.enabled:
            self.set_batch_inspector(BatchInspector(
                log_first=train_cfg.inspector.log_first,
                log_every=train_cfg.inspector.log_every,
            ))


    def fit(self, train_loader: DataLoader, validation_loader: DataLoader | None, test_loader: DataLoader | None = None) -> GwENMultiTrainingResult:
        """Fit the model using the provided data loaders and return training results."""
        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.test_loader = test_loader
        self.train_loss_history.clear()
        self.val_auc_history.clear()

        super().fit_epochs()
        return GwENBinaryTrainingResult(
            train_loss_history=list(self.train_loss_history),
            val_auc_history=list(self.val_metric_history),
            best_auc=self.best_metric or 0.0,
        )
    
    def compute_loss(self, outputs: torch.Tensor, batch: Any) -> torch.Tensor:
        import torch.nn.functional as F
        return F.binary_cross_entropy(outputs, batch["targets"].float())
