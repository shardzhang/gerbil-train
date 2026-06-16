"""Trainer for GwEN CTR (binary classification) models."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from gerbil_train.config import GwENTrainConfig
from gerbil_train.metrics.classification import auc
from gerbil_train.trainer.base_trainer import BaseTrainer
from gerbil_train.utils import BatchInspector
from gerbil_train.utils.plot import save_curve_values

__all__ = ["GwENBinaryTrainer", "GwENBinaryTrainingResult"]


@dataclass
class GwENBinaryTrainingResult:
    train_loss_history: list[float]
    val_auc_history: list[float]
    best_auc: float


class GwENBinaryTrainer(BaseTrainer):
    """Trainer for GwEN binary classification models."""

    def __init__(self, model: nn.Module, config: GwENTrainConfig) -> None:
        optimizer_cfg = config.optimizer
        scheduler_cfg = config.scheduler
        checkpoint_cfg = config.checkpoint
        early_stop_cfg = config.early_stop
        logging_cfg = config.logging

        optimizer = optim.Adam(
            model.parameters(),
            lr=float(optimizer_cfg.lr or 1e-3),
            weight_decay=float(optimizer_cfg.weight_decay or 0.0),
        )

        scheduler = None
        if scheduler_cfg.enabled:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode=str(scheduler_cfg.mode),
                factor=float(scheduler_cfg.factor), patience=int(scheduler_cfg.patience),
            )

        device = config.device or ("cuda" if torch.cuda.is_available() else "cpu")
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"

        super().__init__(
            model=model, optimizer=optimizer, scheduler=scheduler, device=device,
            gradient_clip_norm=None,
            monitor=str(checkpoint_cfg.monitor or "val_auc"),
            monitor_mode=str(checkpoint_cfg.mode or "max"),
            patience=0 if not early_stop_cfg.enabled else int(early_stop_cfg.patience),
            best_checkpoint_path=checkpoint_cfg.path,
            best_metric=None, wait=0, seed=config.seed,
        )

        self.config = config
        self.epochs = int(config.epochs)
        self.train_loader: DataLoader | None = None
        self.validation_loader: DataLoader | None = None
        self.test_loader: DataLoader | None = None
        self.train_loss_history: list[float] = []
        self.val_auc_history: list[float] = []
        self.plot_path = Path(logging_cfg.plot_path) if logging_cfg.plot_path is not None else None

        if config.inspector.enabled:
            self.set_batch_inspector(BatchInspector(
                log_first=config.inspector.log_first,
                log_every=config.inspector.log_every,
            ))

    def fit(
        self, train_loader: DataLoader, validation_loader: DataLoader | None,
        test_loader: DataLoader | None = None,
    ) -> GwENBinaryTrainingResult:
        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.test_loader = test_loader
        self.train_loss_history.clear()
        self.val_auc_history.clear()
        super().fit(epochs=self.epochs)
        return GwENBinaryTrainingResult(
            train_loss_history=list(self.train_loss_history),
            val_auc_history=list(self.val_auc_history),
            best_auc=self.best_metric or 0.0,
        )

    def train_one_epoch(self, epoch: int) -> dict[str, float]:
        if self.train_loader is None:
            return {"loss": 0.0}
        self.model.train()
        total_loss = 0.0
        pbar = tqdm(self.train_loader, total=self.steps_per_epoch or None, desc=f"Epoch {epoch + 1}/{self.max_epochs} [train]", leave=False)
        for step, batch in enumerate(pbar, start=1):
            batch = self.move_batch_to_device(batch)
            self.on_train_step_start(batch)
            self.inspect_batch(step, batch)
            self.zero_grad()
            outputs = self.model(batch["feature_bags"])
            labels = batch["targets"].float()
            loss = F.binary_cross_entropy(outputs, labels)
            self.backward_step(loss)
            self.clip_gradients()
            self.optimizer_step()
            total_loss += float(loss.item())
            pbar.set_postfix(loss=f"{total_loss / step:.4f}")
            self.global_step += 1
        return {"loss": total_loss / max(step, 1)}

    def validate(self, epoch: int | None = None) -> dict[str, float]:
        if self.validation_loader is None:
            return {}
        self.model.eval()
        total_auc = 0.0
        total_loss = 0.0
        total_steps = 0
        with torch.no_grad():
            for batch in self.validation_loader:
                batch = self.move_batch_to_device(batch)
                outputs = self.model(batch["feature_bags"])
                labels = batch["targets"].float()
                total_loss += float(F.binary_cross_entropy(outputs, labels).item())
                total_auc += auc(labels, outputs)
                total_steps += 1
        d = max(total_steps, 1)
        return {"loss": total_loss / d, "auc": total_auc / d}

    def evaluate(self, dataloader: DataLoader | None = None) -> dict[str, float]:
        if dataloader is None:
            return {}
        self.model.eval()
        total_auc = 0.0
        total_steps = 0
        with torch.no_grad():
            for batch in dataloader:
                batch = self.move_batch_to_device(batch)
                outputs = self.model(batch["feature_bags"])
                labels = batch["targets"].float()
                total_auc += auc(labels, outputs)
                total_steps += 1
        return {"test_auc": total_auc / max(total_steps, 1)}

    def forward_step(self, batch: dict[str, Any]) -> torch.Tensor:
        return self.model(batch["feature_bags"])

    def compute_loss(self, outputs: torch.Tensor, batch: dict[str, Any]) -> torch.Tensor:
        return F.binary_cross_entropy(outputs, batch["targets"].float())

    def on_validation_end(self, metrics: dict[str, float]) -> None:
        v = metrics.get("auc")
        if v is not None:
            self.scheduler_step(v)

    def on_epoch_end(self, epoch: int, metrics: dict[str, float]) -> None:
        train_loss = metrics.get("train_loss")
        val_auc = metrics.get("val_auc")
        if train_loss is not None:
            self.train_loss_history.append(float(train_loss))
        if val_auc is not None:
            self.val_auc_history.append(float(val_auc))
        msg = f"Epoch {epoch + 1} | loss: {train_loss:.4f}" if train_loss is not None else f"Epoch {epoch + 1}"
        if val_auc is not None:
            msg += f" | AUC val: {val_auc:.4f}"
        self.finalize_epoch(epoch, metrics, msg)

    def save_training_artifacts(self) -> None:
        if self.plot_path is None:
            return
        self.save_loss_curve()
        self.save_metric_curve()
        self.plot_loss_curve()
        self.plot_metric_curve()

    def save_loss_curve(self) -> None:
        if self.plot_path is None:
            return
        save_curve_values(self.train_loss_history, self.plot_path.with_name(f"{self.plot_path.stem}_loss.txt"))

    def save_metric_curve(self) -> None:
        if self.plot_path is None:
            return
        save_curve_values(self.val_auc_history, self.plot_path.with_name(f"{self.plot_path.stem}_metric.txt"))

    def plot_loss_curve(self) -> None:
        if self.plot_path is None or not self.train_loss_history:
            return
        plt.figure(figsize=(8, 4))
        plt.plot(range(1, len(self.train_loss_history) + 1), self.train_loss_history, label="train_loss")
        plt.xlabel("Epoch"); plt.ylabel("Loss");         plt.title("GwEN Binary Training Loss")
        plt.grid(True, linestyle="--", alpha=0.3); plt.legend(); plt.tight_layout()
        self.plot_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(self.plot_path.with_name(f"{self.plot_path.stem}_loss.png"))
        plt.close()

    def plot_metric_curve(self) -> None:
        if self.plot_path is None or not self.val_auc_history:
            return
        plt.figure(figsize=(8, 4))
        plt.plot(range(1, len(self.val_auc_history) + 1), self.val_auc_history, label="val_auc")
        plt.xlabel("Epoch"); plt.ylabel("AUC");         plt.title("GwEN Binary Validation AUC")
        plt.grid(True, linestyle="--", alpha=0.3); plt.legend(); plt.tight_layout()
        self.plot_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(self.plot_path.with_name(f"{self.plot_path.stem}_metric.png"))
        plt.close()
