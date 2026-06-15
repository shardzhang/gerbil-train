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

from gerbil_train.metrics.ranking import ndcg_score
from gerbil_train.trainer.base_trainer import BaseTrainer
from gerbil_train.utils.plot import save_curve_values

__all__ = ["DeepFMTrainer", "DeepFMTrainingResult"]


@dataclass
class DeepFMTrainingResult:
    """Container for aggregated DeepFM training results."""

    train_loss_history: list[float]
    val_ndcg_history: list[float]
    best_ndcg: float


class DeepFMTrainer(BaseTrainer):
    """Trainer for DeepFM recommendation models."""

    def __init__(self, model: nn.Module, config: dict[str, Any]) -> None:
        """Initialize the DeepFM trainer.

        :param model: DeepFM model instance
        :param config: Training configuration mapping
        """
        optimizer_cfg = config.get("optimizer", {})
        scheduler_cfg = config.get("scheduler", {})
        checkpoint_cfg = config.get("checkpoint", {})
        early_stop_cfg = config.get("early_stop", {})
        logging_cfg = config.get("logging", {})
        evaluation_cfg = config.get("evaluation", {})
        gradient_cfg = config.get("gradient", {})

        optimizer = optim.Adam(
            model.parameters(),
            lr=float(optimizer_cfg.get("lr", 1e-3)),
            weight_decay=float(optimizer_cfg.get("weight_decay", 0.0)),
        )

        scheduler = None
        if scheduler_cfg.get("enabled", False):
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode=str(scheduler_cfg.get("mode", "max")),
                factor=float(scheduler_cfg.get("factor", 0.5)),
                patience=int(scheduler_cfg.get("patience", 2)),
            )

        device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"

        super().__init__(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            gradient_clip_norm=gradient_cfg.get("clip_grad_norm"),
            monitor=str(checkpoint_cfg.get("monitor", "val_ndcg@10")),
            monitor_mode=str(checkpoint_cfg.get("mode", "max")),
            patience=(
                int(early_stop_cfg.get("patience", 3))
                if early_stop_cfg.get("enabled", True)
                else 0
            ),
            best_checkpoint_path=checkpoint_cfg.get("best_checkpoint_path"),
            best_metric=None,
            wait=0,
            seed=config.get("seed", 42),
        )

        self.config = config
        self.epochs = int(config.get("epochs", 5))
        self.validation_k = int(evaluation_cfg.get("validation_k", 10))
        self.train_loader: DataLoader | None = None
        self.validation_loader: DataLoader | None = None
        self.test_loader: DataLoader | None = None
        self.train_loss_history: list[float] = []
        self.val_ndcg_history: list[float] = []
        self.log_dir = logging_cfg.get("log_dir")
        self.plot_path = (
            Path(logging_cfg["plot_path"])
            if logging_cfg.get("plot_path") is not None
            else None
        )

    def fit(
        self,
        train_loader: DataLoader,
        validation_loader: DataLoader | None,
        test_loader: DataLoader | None = None,
    ) -> DeepFMTrainingResult:
        """Run DeepFM training and return aggregated history."""
        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.test_loader = test_loader
        self.train_loss_history.clear()
        self.val_ndcg_history.clear()

        super().fit(epochs=self.epochs)
        return DeepFMTrainingResult(
            train_loss_history=list(self.train_loss_history),
            val_ndcg_history=list(self.val_ndcg_history),
            best_ndcg=self.best_metric or 0.0,
        )

    def train_one_epoch(self, epoch: int) -> dict[str, float]:
        """Train DeepFM for one epoch."""
        if self.train_loader is None:
            return {"loss": 0.0}

        self.model.train()
        total_loss = 0.0
        train_pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch + 1}/{self.max_epochs} [train]",
            leave=False,
        )
        for step, batch in enumerate(train_pbar, start=1):
            batch = self.move_batch_to_device(batch)
            self.zero_grad()
            outputs = self.forward_step(batch)
            loss = self.compute_loss(outputs, batch)
            self.backward_step(loss)
            self.clip_gradients()
            self.optimizer_step()
            total_loss += float(loss.item())
            train_pbar.set_postfix(loss=f"{total_loss / step:.4f}")
            self.global_step += 1

        avg_loss = total_loss / max(len(self.train_loader), 1)
        return {"loss": avg_loss}

    def validate(self, epoch: int | None = None) -> dict[str, float]:
        """Run ranking validation on held-out groups."""
        if self.validation_loader is None:
            return {}

        self.model.eval()
        ndcg_total = 0.0
        total_steps = 0

        with torch.no_grad():
            for batch in self.validation_loader:
                batch = self.move_batch_to_device(batch)
                outputs = self.forward_step(batch)
                ndcg_total += float(
                    ndcg_score(batch["labels"], outputs, k=self.validation_k)
                )
                total_steps += 1

        return {f"ndcg@{self.validation_k}": ndcg_total / max(total_steps, 1)}

    def evaluate(self, dataloader: DataLoader | None = None) -> dict[str, float]:
        """Run ranking evaluation on a held-out test loader."""
        if dataloader is None:
            return {}

        self.model.eval()
        ndcg_total = 0.0
        total_steps = 0
        with torch.no_grad():
            for batch in dataloader:
                batch = self.move_batch_to_device(batch)
                outputs = self.forward_step(batch)
                ndcg_total += float(
                    ndcg_score(batch["labels"], outputs, k=self.validation_k)
                )
                total_steps += 1
        return {f"test_ndcg@{self.validation_k}": ndcg_total / max(total_steps, 1)}

    def forward_step(self, batch: dict[str, Any]) -> torch.Tensor:
        """Run the DeepFM forward pass for one batch."""
        dense_features = batch.get("dense_features")
        if dense_features is not None and dense_features.numel() == 0:
            dense_features = None
        return self.model(
            dense_features=dense_features,
            sparse_features=batch["sparse_features"],
        )

    def compute_loss(self, outputs: torch.Tensor, batch: dict[str, Any]) -> torch.Tensor:
        """Compute the pointwise DeepFM loss."""
        labels = batch["label"].float()
        return F.binary_cross_entropy(outputs, labels)

    def on_epoch_end(self, epoch: int, metrics: dict[str, float]) -> None:
        """Record DeepFM epoch metrics."""
        train_loss = metrics.get("train_loss")
        val_ndcg = metrics.get(f"val_ndcg@{self.validation_k}")
        if train_loss is not None:
            self.train_loss_history.append(float(train_loss))
        if val_ndcg is not None:
            self.val_ndcg_history.append(float(val_ndcg))

        message = (
            f"Epoch {epoch + 1} | loss: {train_loss:.4f}"
            if train_loss is not None
            else f"Epoch {epoch + 1}"
        )
        if val_ndcg is not None:
            message += f" | NDCG@{self.validation_k} val: {val_ndcg:.4f}"
        self.log_message(message)

    def save_training_artifacts(self) -> None:
        """Persist DeepFM training curves when configured."""
        if self.plot_path is None:
            return
        self.save_loss_curve()
        self.save_metric_curve()
        self.plot_loss_curve()
        self.plot_metric_curve()

    def save_loss_curve(self) -> None:
        """Save DeepFM training-loss values to a text file."""
        if self.plot_path is None:
            return
        save_curve_values(
            self.train_loss_history,
            self.plot_path.with_name(f"{self.plot_path.stem}_loss.txt"),
        )

    def save_metric_curve(self) -> None:
        """Save DeepFM validation-NDCG values to a text file."""
        if self.plot_path is None:
            return
        save_curve_values(
            self.val_ndcg_history,
            self.plot_path.with_name(f"{self.plot_path.stem}_metric.txt"),
        )

    def plot_loss_curve(self) -> None:
        """Render DeepFM loss curve figure."""
        if self.plot_path is None:
            return
        plt.figure(figsize=(8, 4))
        plt.plot(
            range(1, len(self.train_loss_history) + 1),
            self.train_loss_history,
            label="train_loss",
        )
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("DeepFM Training Loss")
        plt.grid(True, linestyle="--", alpha=0.3)
        plt.legend()
        plt.tight_layout()
        self.plot_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(self.plot_path.with_name(f"{self.plot_path.stem}_loss.png"))
        plt.close()

    def plot_metric_curve(self) -> None:
        """Render DeepFM validation-NDCG curve figure."""
        if self.plot_path is None or not self.val_ndcg_history:
            return
        plt.figure(figsize=(8, 4))
        plt.plot(
            range(1, len(self.val_ndcg_history) + 1),
            self.val_ndcg_history,
            label=f"val_ndcg@{self.validation_k}",
        )
        plt.xlabel("Epoch")
        plt.ylabel("NDCG")
        plt.title("DeepFM Validation NDCG")
        plt.grid(True, linestyle="--", alpha=0.3)
        plt.legend()
        plt.tight_layout()
        self.plot_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(self.plot_path.with_name(f"{self.plot_path.stem}_metric.png"))
        plt.close()
