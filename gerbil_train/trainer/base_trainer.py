"""Base trainer abstractions and lifecycle hooks."""

from __future__ import annotations

import time

from pathlib import Path
from typing import Any

import math

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader

from gerbil_train.utils.nn import count_parameters, print_model_structure
from gerbil_train.utils.seed import set_seed
from gerbil_train.metrics.classification import auc
from gerbil_train.data.tfrecord_dataset import collect_tfrecord_part_files, count_tfrecord_records

__all__ = ["BaseTrainer"]


class BaseTrainer:
    """Base trainer abstraction for model training, evaluation, and utilities."""
    def __init__(
        self,
        *,
        model: nn.Module,
        optimizer: Optimizer,
        scheduler: LRScheduler | None,
        device: torch.device | str,
        gradient_clip_norm: float | None,
        monitor: str,
        monitor_mode: str,
        patience: int,
        best_checkpoint_path: str | Path | None,
        best_metric: float | None,
        wait: int,
        seed: int | None,
        verbose: bool = False,
    ) -> None:
        """Initialize shared trainer state.

        :param model: Model to train
        :param optimizer: Optimizer used for parameter updates
        :param scheduler: Optional learning rate scheduler
        :param device: Device used for model execution
        :param gradient_clip_norm: Optional gradient clipping threshold
        :param monitor: Metric name used for checkpointing and early stopping
        :param monitor_mode: Whether a lower or higher monitor value is better
        :param patience: Early stopping patience measured in epochs
        :param best_checkpoint_path: Optional destination path for the best checkpoint
        :param best_metric: Initial best metric value
        :param wait: Initial early stopping wait counter
        :param seed: Optional random seed for reproducibility
        :param verbose: Whether to print training progress
        """
        self.model: nn.Module = model
        self.optimizer: Optimizer = optimizer
        self.scheduler: LRScheduler | None = scheduler
        self.device: torch.device = torch.device(device)
        self.gradient_clip_norm: float | None = gradient_clip_norm
        # 不同任务的monitor不同，例如分类任务是auc/gauc，回归任务是mse等
        self.monitor: str = monitor
        # 不同任务的monitor_mode不同，例如分类任务是auc/gauc的max，回归任务是mse的min
        self.monitor_mode: str = monitor_mode
        self.patience: int = patience
        self.best_checkpoint_path: Path | None = Path(best_checkpoint_path)
        # best_metric是monitor的具体值
        self.best_metric: float | None = best_metric
        self.wait: int = wait
        self.seed: int | None = seed
        self.verbose: bool = verbose

        self.model_name: str = "Model"
        self.current_epoch: int = 0
        self.epochs: int = 0
        self.global_step: int = 0

        self.plot_path: Path | None = None
        self.train_loss_history: list[float] = []
        self.val_loss_history: list[float] = []

        self._batch_inspector = None
        self._total_train_samples: int = 0
        self._steps_per_epoch: int = 0
        self._profile_path: Path | None = None
        self._epoch_start_time: float = 0.0

        self._initial_lr: float = 0.0
        self._scheduler_cfg: Any = None

        if self.best_checkpoint_path is not None:
            self.set_profile_path(self.best_checkpoint_path)


    def set_batch_inspector(self, inspector: Any) -> None:
        """Attach a batch inspector for logging training batches."""
        self._batch_inspector = inspector


    def setup_total_train_samples(self, data_cfg: dict, batch_size: int, split: str = "train") -> None:
        """Compute total training samples from data config and set on trainer."""
        root_dir = Path(data_cfg["paths"]["tfrecord_root"]) / data_cfg.get("split_subdirs", {}).get(split, split) / "tfrecord"
        train_files = collect_tfrecord_part_files(root_dir)
        self._total_train_samples = count_tfrecord_records(train_files)
        self._steps_per_epoch = (self._total_train_samples + batch_size - 1) // batch_size


    def inspect_batch(self, step: int, batch: Any) -> None:
        """Inspect a training batch if a batch inspector is attached.
        Call this inside ``train_one_epoch`` at the desired step.
        """
        if self._batch_inspector is not None:
            self._batch_inspector(step, batch, self.current_epoch)


    def set_profile_path(self, run_dir: Path) -> None:
        """Set the path for saving training profile logs."""
        self._profile_path = run_dir / "profile.txt"


    def finalize_epoch(self, epoch: int, metrics: dict[str, float], message: str) -> None:
        """Finalize epoch by logging a message with elapsed time and steps per second."""
        elapsed = time.time() - self._epoch_start_time
        steps_per_sec = self._steps_per_epoch / elapsed if elapsed > 0 and self._steps_per_epoch > 0 else 0.0
        message += f" | steps: {self._steps_per_epoch} | steps/s: {steps_per_sec:.2f} | time: {elapsed:.1f}s"
        self.log_message(message)
        if self._profile_path is not None:
            self._profile_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._profile_path, "a") as f:
                f.write(message + "\n")


    def on_fit_start(self, train_loader: Any, val_loader: Any, test_loader: Any) -> None:
        """Hook called at the start of fit() with the prepared dataloaders."""
        self.train_loader = train_loader
        self.validation_loader = val_loader
        self.test_loader = test_loader
        self.train_loss_history.clear()
        pass


    def fit(self, train_loader: Any, val_loader: Any = None, test_loader: Any = None) -> None:
        """Run the full training lifecycle with the given dataloaders."""
        self.on_fit_start(train_loader, val_loader, test_loader)
        self.fit_epochs()
        raise NotImplementedError()
    

    def fit_epochs(self) -> None:
        """Run the full training lifecycle."""
        self.setup()
        self.on_train_start()

        if self._total_train_samples > 0:
            self.log_message(f"Train samples: {self._total_train_samples} | Steps/epoch: {self._steps_per_epoch}")

        try:
            for epoch in range(self.epochs):
                self.current_epoch = epoch
                self.on_epoch_start(epoch)

                train_metrics = self.train_one_epoch(epoch)
                metrics = {
                    f"train_{key}": value 
                    for key, value in train_metrics.items()
                }

                val_metrics = self.validate(epoch)
                metrics.update(
                    {f"val_{key}": value 
                    for key, value in val_metrics.items()}
                )
                self.on_validation_end(metrics)
                self.on_epoch_end(epoch, metrics)
                if self.update_best_state(metrics):
                    break
        finally:
            self.on_train_end()
            self.cleanup()


    def setup(self) -> None:
        """Prepare trainer state before training starts.

        The default implementation applies the configured random seed if present.
        Subclasses can extend this to create loggers, writers, directories, or caches.
        """
        if self.seed is not None:
            self.log_message(f"Setting random seed to {self.seed}")
            set_seed(self.seed)

        self.model.to(self.device)

        if self.verbose:
            self.log_message(f"verbose: {self.verbose}")
            print_model_structure(self.model)
            count_parameters(self.model)

        if self.best_checkpoint_path is not None:
            self.log_message(f"Setting best checkpoint path to {self.best_checkpoint_path}")
            self.best_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        if self.plot_path is not None and not self.plot_path.suffix:
            self.log_message(f"Setting plot path to {self.plot_path}")
            self.plot_path = self.plot_path / "training_curves"


    def cleanup(self) -> None:
        """Release resources after training completes."""
        pass


    def on_train_start(self) -> None:
        """Hook called before the first training epoch."""
        self.log_message(f"Begin training for {self.model_name}")
        pass


    def on_epoch_start(self, epoch: int) -> None:
        """Hook called before each epoch starts.
        :param epoch: Zero-based epoch index
        """
        self._epoch_start_time = time.time()


    def train_one_epoch(self, epoch: int) -> dict[str, float]:
        """ Run one epoch of training and return aggregated metrics."""
        raise NotImplementedError


    def validate(self, epoch: int | None = None) -> dict[str, float]:
        """Run validation and return aggregated metrics."""
        raise NotImplementedError


    def on_validation_end(self, metrics: dict[str, float]) -> None:
        """Hook called after validation ends. LR is updated per-step via update_learning_rate()."""
        pass

    def on_epoch_end(self, epoch: int, metrics: dict[str, float]) -> None:
        """ Hook called after each epoch ends.
        :param epoch: Zero-based epoch index
        :param metrics: Aggregated metrics for the current epoch
        """
        raise NotImplementedError


    def on_train_end(self) -> None:
        """Hook called once after the training loop ends.
        The default implementation persists training artifacts such as saved
        curve values and plots.
        """
        self.save_training_artifacts()


    def save_training_artifacts(self) -> None:
        pass


    def evaluate(self, dataloader: DataLoader | None = None) -> dict[str, float]:
        """Run evaluation on the given dataloader and return metrics."""
        raise NotImplementedError


    def on_train_step_start(self, batch: Any) -> None:
        """Hook called before each training step."""
        pass


    def on_train_step_end(self, metrics: dict[str, float]) -> None:
        """Hook called after each training step."""
        pass


    def zero_grad(self) -> None:
        """Reset gradients for all model parameters."""
        self.optimizer.zero_grad(set_to_none=True)


    def forward_step(self, batch: Any) -> Any:
        """Run a forward pass of the model for one training step."""
        raise NotImplementedError


    def compute_loss(self, logits: torch.Tensor, targets: Any) -> torch.Tensor:
        """Compute the loss value for a training step."""
        raise NotImplementedError


    def compute_metrics(self, outputs: torch.Tensor, batch: Any) -> dict[str, float]:
        """Compute metrics from model outputs and label tensors."""
        raise NotImplementedError


    def backward_step(self, loss: torch.Tensor) -> None:
        """Run backpropagation for one loss value."""
        loss.backward()


    def clip_gradients(self) -> None:
        """Clip gradients when a max norm is configured."""
        if self.gradient_clip_norm is None:
            return
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            max_norm=self.gradient_clip_norm,
        )


    def optimizer_step(self) -> None:
        """Apply one optimizer update."""
        self.optimizer.step()


    def scheduler_step(self, step: int) -> None:
        """Step-level learning rate scheduling via ``_scheduler_cfg.type``.

        - ``warmup_exp_decay``: linear warmup + exponential decay
        - ``warmup_cos_decay``: linear warmup + cosine decay

        - exp decay: 起始下降快，后期趋平，适合快速收敛到最优区域
        - cos decay: 平滑下降，无突变，在 Transformer 和推荐模型中更常用
        :param step: Current global step (0-indexed)
        """
        if not hasattr(self, "_scheduler_cfg"):
            return

        cfg = self._scheduler_cfg
        warmup = cfg.warmup_steps
        lr_min = cfg.learning_rate_min
        if cfg.type not in ("warmup_exp_decay", "warmup_cos_decay"):
            return

        # Phase 1: linear warmup (shared by both types)
        lr = self._initial_lr
        if warmup > 0 and step < warmup:
            lr = lr * (step + 1.0) / warmup
            for group in self.optimizer.param_groups:
                group["lr"] = lr
            return

        # Phase 2: decay after warmup
        if cfg.type == "warmup_exp_decay":
            if cfg.decay_rate < 0:
                lr = self._initial_lr * math.exp(cfg.decay_rate * (step + 1 - warmup) / max(warmup, 1))
        elif cfg.type == "warmup_cos_decay":
            total = max(cfg.total_steps, warmup + 1)
            progress = (step - warmup) / (total - warmup)
            lr = lr_min + 0.5 * (self._initial_lr - lr_min) * (1 + math.cos(math.pi * max(0, min(progress, 1))))

        lr = max(lr, lr_min)
        for group in self.optimizer.param_groups:
            group["lr"] = lr


    def on_evaluate_start(self) -> None:
        """Hook called before evaluation starts."""
        pass


    def on_evaluate_end(self, metrics: dict[str, float]) -> None:
        """Hook called after evaluation ends.
        :param metrics: Evaluation metrics
        """
        pass


    def predict(self, *args: Any, **kwargs: Any) -> dict[str, float]:
        """Run prediction outside the training loop."""
        # self.on_predict_start hook
        # self.forward_step
        # self.on_predict_end hook
        raise NotImplementedError


    def on_predict_start(self) -> None:
        """Hook called before prediction starts."""
        pass


    def on_predict_end(self) -> None:
        """Hook called after prediction ends."""
        pass


    def save_checkpoint(self, path: str | Path) -> None:
        """Save model, optimizer, and trainer state.

        :param path: Destination checkpoint path
        """
        checkpoint_path = Path(path)
        if checkpoint_path.is_dir():
            checkpoint_path = checkpoint_path / "best_model.pth"
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_metric": self.best_metric,
            "wait": self.wait,
            "current_epoch": self.current_epoch,
            "global_step": self.global_step,
        }

        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()
        torch.save(checkpoint, checkpoint_path)


    def load_checkpoint(self, path: str | Path) -> None:
        """Load model, optimizer, and trainer state.
        :param path: Source checkpoint path
        """
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])
            if "optimizer_state_dict" in checkpoint:
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

            if self.scheduler is not None and "scheduler_state_dict" in checkpoint:
                self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

            self.best_metric = checkpoint.get("best_metric", self.best_metric)
            self.wait = checkpoint.get("wait", self.wait)
            self.current_epoch = checkpoint.get("current_epoch", self.current_epoch)
            self.global_step = checkpoint.get("global_step", self.global_step)
            return
        self.model.load_state_dict(checkpoint)


    def update_best_state(self, metrics: dict[str, float]) -> bool:
        """Check whether the monitored metric improved.

        The default logic compares the current monitored metric with the best
        value seen so far, saves the best checkpoint when the metric improves,
        and stops training when the metric fails to improve for ``patience``
        consecutive checks.

        :param metrics: Aggregated metrics for the current epoch
        :return: ``True`` when training should stop early
        """
        monitored_value = metrics.get(self.monitor)
        if monitored_value is None:
            return False

        if self.is_better(monitored_value):
            self.best_metric = monitored_value
            self.wait = 0
            if self.best_checkpoint_path is not None:
                self.save_checkpoint(self.best_checkpoint_path)
                self.log_message(f"Saved best model to {self.best_checkpoint_path.resolve()}")
            return False

        self.wait += 1
        if self.patience > 0 and self.wait >= self.patience:
            self.log_message(
                f"Early stopping at epoch {self.current_epoch + 1}: "
                f"{self.monitor} did not improve for {self.patience} epoch(s)."
            )
            return True
        return False


    def is_better(self, value: float) -> bool:
        """Compare a scalar metric using the configured monitor mode.

        :param value: Current metric value
        :return: Whether ``value`` improves on ``self.best_metric``
        """
        if self.best_metric is None:
            return True

        if self.monitor_mode == "min":
            return value < self.best_metric
        return value > self.best_metric


    def move_batch_to_device(self, batch: Any) -> Any:
        """Move tensors in a batch structure to the trainer device.

        :param batch: Input batch structure
        :return: Batch structure with tensor values moved to ``self.device``
        """
        if isinstance(batch, torch.Tensor):
            return batch.to(self.device)

        if isinstance(batch, dict):
            return {
                key: self.move_batch_to_device(value) 
                for key, value in batch.items()
            }

        if isinstance(batch, list):
            return [self.move_batch_to_device(value) for value in batch]

        if isinstance(batch, tuple):
            return tuple(self.move_batch_to_device(value) for value in batch)

        return batch


    def log_message(self, message: str) -> None:
        """Emit a trainer log message to stdout (also captured by exp.log via _Tee)."""
        print(message)
