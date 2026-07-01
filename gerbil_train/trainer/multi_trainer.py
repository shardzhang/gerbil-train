"""Shared trainer for multi-class classification models."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from tqdm.auto import tqdm

import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader

from gerbil_train.utils import BatchInspector
from gerbil_train.utils.plot import save_curve_values
from gerbil_train.config.train_config import TrainConfig
from gerbil_train.losses.classification import nce_loss, sampled_softmax_loss
from gerbil_train.metrics.classification import hit_rate
from gerbil_train.trainer.base_trainer import BaseTrainer


class MultiClassClassificationTrainer(BaseTrainer):
    """Shared trainer for multi-class classification models.

    Provides the full training lifecycle: train/validate/evaluate with configurable
    loss (CE/NCE/SampledSoftmax), Hit@k metrics, checkpointing, and training curves.
    Subclasses only need to set ``model_name`` and optionally override
    ``forward_step`` / ``compute_loss``.
    """
    def __init__(self, model: nn.Module, train_cfg: TrainConfig, data_cfg: dict[str, Any] | None = None) -> None:
        optimizer_cfg = train_cfg.optimizer
        scheduler_cfg = train_cfg.scheduler
        checkpoint_cfg = train_cfg.checkpoint
        early_stop_cfg = train_cfg.early_stop
        logging_cfg = train_cfg.logging

        lr = optimizer_cfg.lr
        optimizer = optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=optimizer_cfg.weight_decay,
        )
        self._initial_lr = float(lr)
        self._scheduler_cfg = scheduler_cfg

        scheduler = None
        if scheduler_cfg.enabled:
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode=scheduler_cfg.mode,
                factor=scheduler_cfg.factor,
                patience=scheduler_cfg.patience,
            )

        super().__init__(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            device=train_cfg.device or ("cuda" if torch.cuda.is_available() else "cpu"),
            gradient_clip_norm=None,
            monitor=checkpoint_cfg.monitor or "val_hit@1",
            monitor_mode=checkpoint_cfg.mode or "max",
            patience=0 if not early_stop_cfg.enabled else int(early_stop_cfg.patience),
            best_checkpoint_path=checkpoint_cfg.path,
            best_metric=None,
            wait=0,
            seed=train_cfg.seed,
            verbose=logging_cfg.verbose,
        )

        self.model_name = "MultiClassModel"
        self.config = train_cfg
        self.epochs = int(train_cfg.epochs)
        self.loss_cfg = train_cfg.loss
        self.train_loader: DataLoader | None = None
        self.validation_loader: DataLoader | None = None
        self.test_loader: DataLoader | None = None

        self.plot_path = Path(logging_cfg.plot_path)
        self.train_loss_history: list[float] = []
        self.val_loss_history: list[float] = []
        self.val_hit1_history: list[float] = []
        self.val_hit10_history: list[float] = []

        if data_cfg is not None:
            self.setup_total_train_samples(data_cfg, train_cfg.data.batch_size)

        if train_cfg.inspector.enabled:
            self.set_batch_inspector(BatchInspector(
                log_first=train_cfg.inspector.log_first,
                log_every=train_cfg.inspector.log_every,
            ))

    def fit(self, train_loader: DataLoader, validation_loader: DataLoader | None, test_loader: DataLoader | None = None) -> None:
        """Fit the model using the provided data loaders."""
        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.test_loader = test_loader

        self.train_loss_history.clear()
        self.val_loss_history.clear()
        self.val_hit1_history.clear()
        self.val_hit10_history.clear()

        super().fit_epochs()

    def train_one_epoch(self, epoch: int) -> dict[str, float]:
        """Train for one epoch and return average loss."""
        if self.train_loader is None:
            return {"loss": 0.0}

        self.model.train()
        total_loss: float = 0.0
        total_steps: int = 0
        train_pbar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{self.epochs} [train]", leave=False)
        for step, batch in enumerate(train_pbar, start=1):
            batch = self.move_batch_to_device(batch)
            self.inspect_batch(step, batch)
            if self.loss_cfg.type in ("nce", "sampled_softmax"):
                hidden = self.model.encode(batch["feature_bags"])
                if self.loss_cfg.type == "nce":
                    loss = nce_loss(hidden, self.model.head.weight, batch["targets"].long(), num_sampled=self.loss_cfg.num_sampled)
                elif self.loss_cfg.type == "sampled_softmax":
                    loss = sampled_softmax_loss(hidden, self.model.head.weight, batch["targets"].long(), num_sampled=self.loss_cfg.num_sampled, class_bias=self.model.head.bias)
                else:
                    raise NotImplementedError(f"Loss type {self.loss_cfg.type} not implemented")
            elif self.loss_cfg.type == "ce":
                logits = self.forward_step(batch)
                targets = batch["targets"].long()
                loss = self.compute_loss(logits, targets)
            else:
                raise NotImplementedError(f"Loss type {self.loss_cfg.type} not implemented")
            self.zero_grad()
            self.backward_step(loss)
            self.clip_gradients()
            self.optimizer_step()
            total_loss += float(loss.item())
            total_steps += 1
            train_pbar.set_postfix(loss=f"{total_loss / step:.4f}")
            self.global_step += 1
            self.update_learning_rate(self.global_step)
        avg_loss = total_loss / max(total_steps, 1)
        return {"loss": avg_loss}

    def forward_step(self, batch: dict[str, Any]) -> torch.Tensor:
        """Forward pass: return logits."""
        return self.model(batch["feature_bags"])

    def compute_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute cross-entropy loss."""
        return F.cross_entropy(logits, targets)

    def on_epoch_end(self, epoch: int, metrics: dict[str, float]) -> None:
        """Log metrics and build message for finalize_epoch."""
        train_loss = metrics.get("train_loss")
        val_loss = metrics.get("val_loss")
        val_hit1 = metrics.get("val_hit@1")
        val_hit10 = metrics.get("val_hit@10")

        if train_loss is not None:
            self.train_loss_history.append(float(train_loss))
        if val_loss is not None:
            self.val_loss_history.append(float(val_loss))
        if val_hit1 is not None:
            self.val_hit1_history.append(float(val_hit1))
        if val_hit10 is not None:
            self.val_hit10_history.append(float(val_hit10))

        message = f"Epoch {epoch + 1} | loss: {train_loss:.4f}" if train_loss is not None else f"Epoch {epoch + 1}"
        if val_loss is not None:
            message += f" | val_loss: {val_loss:.4f}"
        if val_hit1 is not None:
            message += f" | Hit@1 val: {val_hit1:.4f}"
        if val_hit10 is not None:
            message += f" | Hit@10 val: {val_hit10:.4f}"
        self.finalize_epoch(epoch, metrics, message)

    @torch.no_grad()
    def validate(self, epoch: int | None = None) -> dict[str, float]:
        """Evaluate on validation set: loss, Hit@1, Hit@10."""
        if self.validation_loader is None:
            return {}

        self.model.eval()
        total_hit1 = 0.0
        total_hit10 = 0.0
        total_loss = 0.0
        total_steps = 0
        for batch in self.validation_loader:
            batch = self.move_batch_to_device(batch)
            logits = self.forward_step(batch)
            targets = batch["targets"].long()
            total_loss += self.compute_loss(logits, targets).item()
            total_hit1 += hit_rate(logits, targets, k=1)
            total_hit10 += hit_rate(logits, targets, k=10)
            total_steps += 1

        denominator = max(total_steps, 1)
        return {
            "loss": total_loss / denominator,
            "val_hit@1": round(total_hit1 / denominator, 4),
            "val_hit@10": round(total_hit10 / denominator, 4),
        }

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader | None = None) -> dict[str, float]:
        """Evaluate on test set: Hit@1, Hit@10."""
        if dataloader is None:
            return {}

        self.model.eval()
        total_hit1 = 0.0
        total_hit10 = 0.0
        total_steps = 0
        for batch in dataloader:
            batch = self.move_batch_to_device(batch)
            logits = self.forward_step(batch)
            targets = batch["targets"].long()
            total_hit1 += hit_rate(logits, targets, k=1)
            total_hit10 += hit_rate(logits, targets, k=10)
            total_steps += 1
        denominator = max(total_steps, 1)
        return {
            "test_hit@1": round(total_hit1 / denominator, 4),
            "test_hit@10": round(total_hit10 / denominator, 4),
        }

    def save_training_artifacts(self) -> None:
        """Save loss and metric curves after training completes."""
        if self.plot_path is None:
            return
        save_curve_values(self.train_loss_history, self.plot_path.with_name(f"{self.plot_path.stem}_loss.txt"))
        save_curve_values(self.val_hit1_history, self.plot_path.with_name(f"{self.plot_path.stem}_metric.txt"))
        self.plot_loss_curve()
        self.plot_metric_curve()

    def plot_loss_curve(self) -> None:
        if self.plot_path is None or not self.train_loss_history:
            return
        from matplotlib import pyplot as plt
        fig, ax1 = plt.subplots(figsize=(8, 4))
        epochs = range(1, len(self.train_loss_history) + 1)

        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Train Loss", color="tab:blue")
        ax1.plot(epochs, self.train_loss_history, color="tab:blue", linestyle="-", label="train_loss")
        ax1.tick_params(axis="y", labelcolor="tab:blue")
        ax1.legend(loc="upper left")

        if self.val_loss_history:
            ax2 = ax1.twinx()
            ax2.set_ylabel("Val Loss", color="tab:red")
            ax2.plot(epochs, self.val_loss_history, color="tab:red", linestyle="--", label="val_loss")
            ax2.tick_params(axis="y", labelcolor="tab:red")
            ax2.legend(loc="upper right")

        plt.title(f"{self.model_name} Loss")
        fig.tight_layout()
        self.plot_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(self.plot_path.with_name(f"{self.plot_path.stem}_loss.png"))
        plt.close()

    def plot_metric_curve(self) -> None:
        if self.plot_path is None or not self.val_hit1_history:
            return
        from matplotlib import pyplot as plt
        fig, ax1 = plt.subplots(figsize=(8, 4))
        epochs = range(1, len(self.val_hit1_history) + 1)

        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Hit@1", color="tab:orange")
        ax1.plot(epochs, self.val_hit1_history, color="tab:orange", linestyle="-", label="Hit@1")
        ax1.tick_params(axis="y", labelcolor="tab:orange")
        ax1.legend(loc="upper left")

        if self.val_hit10_history:
            ax2 = ax1.twinx()
            ax2.set_ylabel("Hit@10", color="tab:green")
            ax2.plot(epochs, self.val_hit10_history, color="tab:green", linestyle="--", label="Hit@10")
            ax2.tick_params(axis="y", labelcolor="tab:green")
            ax2.legend(loc="upper right")

        plt.title(f"{self.model_name} Validation Hit Rate")
        fig.tight_layout()
        self.plot_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(self.plot_path.with_name(f"{self.plot_path.stem}_metric.png"))
        plt.close()
