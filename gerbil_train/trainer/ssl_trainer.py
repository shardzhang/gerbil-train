"""Trainer for SSL contrastive learning models."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn, optim

from gerbil_train.config.train_config import TrainConfig
from gerbil_train.trainer.base_trainer import BaseTrainer

__all__ = ["SSLTrainer"]


class SSLTrainer(BaseTrainer):
    def __init__(self, model: nn.Module, train_cfg: TrainConfig, data_cfg: dict[str, Any] | None = None) -> None:
        optimizer_cfg = train_cfg.optimizer
        checkpoint_cfg = train_cfg.checkpoint
        early_stop_cfg = train_cfg.early_stop
        logging_cfg = train_cfg.logging

        optimizer = optim.Adam(model.parameters(), lr=optimizer_cfg.lr, weight_decay=optimizer_cfg.weight_decay)

        super().__init__(
            model=model,
            optimizer=optimizer,
            scheduler=None,
            device=train_cfg.device or ("cuda" if torch.cuda.is_available() else "cpu"),
            gradient_clip_norm=None,
            monitor="val_loss",
            monitor_mode="min",
            patience=0 if not early_stop_cfg.enabled else int(early_stop_cfg.patience),
            best_checkpoint_path=checkpoint_cfg.path,
            best_metric=None,
            wait=0,
            seed=train_cfg.seed,
            verbose=logging_cfg.verbose,
        )

        self.model_name = "SSL"
        self.epochs = int(train_cfg.epochs)

    def forward_step(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        from gerbil_train.utils.embedding import bag_to_padded

        feature_bags = batch["feature_bags"]
        device = next(self.model.parameters()).device

        item_field = self.model.item_field
        padded_ids, padded_weights, lengths, max_seq_len = bag_to_padded(feature_bags[item_field], device)

        outputs = self.model(padded_ids, lengths)
        loss = self.model.compute_contrastive_loss(outputs["view1"], outputs["view2"])
        return {"loss": loss}

    def compute_loss(self, outputs, batch):
        return outputs["loss"]

    def compute_metrics(self, outputs, batch):
        return {"loss": float(outputs["loss"].item())}

    def fit(self, train_loader, val_loader=None, test_loader=None):
        super().fit(epochs=self.epochs, train_dataloader=train_loader, val_dataloader=val_loader, test_dataloader=test_loader)

    @torch.no_grad()
    def evaluate(self, dataloader=None):
        return {}
