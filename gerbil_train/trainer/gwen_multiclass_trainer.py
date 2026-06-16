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
from gerbil_train.losses.classification import nce_loss, sampled_softmax_loss
from gerbil_train.metrics.classification import hit_rate
from gerbil_train.trainer.base_trainer import BaseTrainer
from gerbil_train.utils import BatchInspector
from gerbil_train.utils.plot import save_curve_values

__all__ = ["GwENTrainer", "GwENTrainingResult"]


@dataclass
class GwENTrainingResult:
    """Container for aggregated GwEN training results."""

    train_loss_history: list[float]
    val_top1_history: list[float]
    best_top1: float


class GwENTrainer(BaseTrainer):
    """Trainer for GwEN multi-class recommendation models."""

    def __init__(self, model: nn.Module, config: GwENTrainConfig) -> None:
        optimizer_cfg = config.optimizer
        scheduler_cfg = config.scheduler
        checkpoint_cfg = config.checkpoint
        early_stop_cfg = config.early_stop
        logging_cfg = config.logging
        evaluation_cfg = config.evaluation

        optimizer_type = str(optimizer_cfg.type or "adam").lower()
        if optimizer_type == "adam":
            optimizer = optim.Adam(
                model.parameters(),
                lr=float(optimizer_cfg.lr or 1e-3),
                weight_decay=float(optimizer_cfg.weight_decay or 0.0),
            )
        else:
            raise ValueError(f"Unsupported optimizer type: {optimizer_type}")

        scheduler = None
        if scheduler_cfg.enabled:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode=str(scheduler_cfg.mode),
                factor=float(scheduler_cfg.factor),
                patience=int(scheduler_cfg.patience),
            )

        device = config.device or ("cuda" if torch.cuda.is_available() else "cpu")
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"

        super().__init__(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            gradient_clip_norm=None,
            monitor=str(checkpoint_cfg.monitor or "val_top1"),
            monitor_mode=str(checkpoint_cfg.mode or "max"),
            patience=0 if not early_stop_cfg.enabled else int(early_stop_cfg.patience),
            best_checkpoint_path=checkpoint_cfg.path,
            best_metric=None,
            wait=0,
            seed=config.seed,
        )

        self.config = config
        self.epochs = int(config.epochs)
        self.topk = int(evaluation_cfg.topk)
        self.loss_cfg = config.loss
        self._use_sampled_loss = self.loss_cfg.type in ("nce", "sampled_softmax")
        self.train_loader: DataLoader | None = None
        self.validation_loader: DataLoader | None = None
        self.test_loader: DataLoader | None = None
        self.train_loss_history: list[float] = []
        self.val_top1_history: list[float] = []
        self.plot_path = Path(logging_cfg.plot_path) if logging_cfg.plot_path is not None else None
        if config.inspector.enabled:
            self.set_batch_inspector(BatchInspector(
                log_first=config.inspector.log_first,
                log_every=config.inspector.log_every,
            ))
        self._single_class_mode = bool(getattr(self.model, "target_size", 0) <= 1)
        if self._single_class_mode:
            self.log_message(
                "GwEN single-class smoke mode enabled: only one target class found. "
                "Loss is set to 0 and metrics are reported as 1.0 for pipeline validation."
            )

    def fit(self, train_loader: DataLoader, validation_loader: DataLoader | None, test_loader: DataLoader | None = None) -> GwENTrainingResult:
        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.test_loader = test_loader
        self.train_loss_history.clear()
        self.val_top1_history.clear()

        super().fit(epochs=self.epochs)
        return GwENTrainingResult(
            train_loss_history=list(self.train_loss_history),
            val_top1_history=list(self.val_top1_history),
            best_top1=self.best_metric or 0.0,
        )

    def train_one_epoch(self, epoch: int) -> dict[str, float]:
        if self.train_loader is None:
            return {"loss": 0.0}

        self.model.train()
        total_loss = 0.0
        total_steps = 0
        train_pbar = tqdm(
            self.train_loader, total=self.steps_per_epoch or None,
            desc=f"Epoch {epoch + 1}/{self.max_epochs} [train]",
            leave=False,
        )
        for step, batch in enumerate(train_pbar, start=1):
            batch = self.move_batch_to_device(batch)
            self.on_train_step_start(batch)
            self.inspect_batch(step, batch)
            if self._use_sampled_loss:
                hidden = self.model.encode(batch["feature_bags"])
                if self.loss_cfg.type == "nce":
                    loss = nce_loss(
                        hidden, self.model.head.weight, batch["targets"].long(),
                        num_sampled=self.loss_cfg.num_sampled,
                    )
                else:
                    loss = sampled_softmax_loss(
                        hidden, self.model.head.weight, batch["targets"].long(),
                        num_sampled=self.loss_cfg.num_sampled,
                        class_bias=self.model.head.bias,
                    )
            else:
                outputs = self.model(batch["feature_bags"])
                loss = F.cross_entropy(outputs, batch["targets"].long())

            self.zero_grad()
            self.backward_step(loss)
            self.clip_gradients()
            self.optimizer_step()
            total_loss += float(loss.item())
            total_steps += 1
            train_pbar.set_postfix(loss=f"{total_loss / step:.4f}")
            self.global_step += 1

        avg_loss = total_loss / max(total_steps, 1)
        return {"loss": avg_loss}

    def validate(self, epoch: int | None = None) -> dict[str, float]:
        if self.validation_loader is None:
            return {}

        self.model.eval()
        total_top1 = 0.0
        total_topk_hit = 0.0
        total_loss = 0.0
        total_steps = 0

        with torch.no_grad():
            for batch in self.validation_loader:
                batch = self.move_batch_to_device(batch)
                logits = self.forward_step(batch)
                targets = batch["targets"].long()

                if logits.size(1) > 1:
                    total_loss += float(F.cross_entropy(logits, targets).item())
                total_top1 += hit_rate(logits, targets, k=1)
                total_topk_hit += hit_rate(logits, targets, self.topk)
                total_steps += 1

        denominator = max(total_steps, 1)
        return {
            "loss": total_loss / denominator,
            "hit@1": total_top1 / denominator,
            f"hit@{self.topk}": total_topk_hit / denominator,
        }

    def evaluate(self, dataloader: DataLoader | None = None) -> dict[str, float]:
        if dataloader is None:
            return {}

        self.model.eval()
        total_top1 = 0.0
        total_topk_hit = 0.0
        total_steps = 0
        with torch.no_grad():
            for batch in dataloader:
                batch = self.move_batch_to_device(batch)
                logits = self.forward_step(batch)
                targets = batch["targets"].long()

                total_top1 += hit_rate(logits, targets, k=1)
                total_topk_hit += hit_rate(logits, targets, self.topk)
                total_steps += 1

        denominator = max(total_steps, 1)
        return {
            "test_hit@1": round(total_top1 / denominator, 4),
            f"test_hit@{self.topk}": round(total_topk_hit / denominator, 4),
        }

    def forward_step(self, batch: dict[str, Any]) -> torch.Tensor:
        return self.model(batch["feature_bags"])

    def compute_loss(self, outputs: torch.Tensor, batch: dict[str, Any]) -> torch.Tensor:
        targets = batch["targets"].long()
        if outputs.size(1) <= 1:
            return outputs.sum() * 0.0
        return F.cross_entropy(outputs, targets)

    def on_validation_end(self, metrics: dict[str, float]) -> None:
        monitored = metrics.get("hit@1")
        if monitored is not None:
            self.scheduler_step(monitored)

    def on_epoch_end(self, epoch: int, metrics: dict[str, float]) -> None:
        train_loss = metrics.get("train_loss")
        val_hit1 = metrics.get("val_hit@1")
        val_hitk = metrics.get(f"val_hit@{self.topk}")

        if train_loss is not None:
            self.train_loss_history.append(float(train_loss))
        if val_hit1 is not None:
            self.val_top1_history.append(float(val_hit1))

        message = (
            f"Epoch {epoch + 1} | loss: {train_loss:.4f}"
            if train_loss is not None
            else f"Epoch {epoch + 1}"
        )
        if val_hit1 is not None:
            message += f" | Hit@1 val: {val_hit1:.4f}"
        if val_hitk is not None:
            message += f" | Hit@{self.topk} val: {val_hitk:.4f}"
        self.finalize_epoch(epoch, metrics, message)

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
        save_curve_values(
            self.train_loss_history,
            self.plot_path.with_name(f"{self.plot_path.stem}_loss.txt"),
        )

    def save_metric_curve(self) -> None:
        if self.plot_path is None:
            return
        save_curve_values(
            self.val_top1_history,
            self.plot_path.with_name(f"{self.plot_path.stem}_metric.txt"),
        )

    def plot_loss_curve(self) -> None:
        if self.plot_path is None or not self.train_loss_history:
            return
        plt.figure(figsize=(8, 4))
        plt.plot(
            range(1, len(self.train_loss_history) + 1),
            self.train_loss_history,
            label="train_loss",
        )
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("GwEN Training Loss")
        plt.grid(True, linestyle="--", alpha=0.3)
        plt.legend()
        plt.tight_layout()
        self.plot_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(self.plot_path.with_name(f"{self.plot_path.stem}_loss.png"))
        plt.close()

    def plot_metric_curve(self) -> None:
        if self.plot_path is None or not self.val_top1_history:
            return
        plt.figure(figsize=(8, 4))
        plt.plot(
            range(1, len(self.val_top1_history) + 1),
            self.val_top1_history,
            label="val_top1",
        )
        plt.xlabel("Epoch")
        plt.ylabel("Top1")
        plt.title("GwEN Validation Top1")
        plt.grid(True, linestyle="--", alpha=0.3)
        plt.legend()
        plt.tight_layout()
        self.plot_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(self.plot_path.with_name(f"{self.plot_path.stem}_metric.png"))
        plt.close()
