"""Base trainer abstractions and lifecycle hooks."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler, ReduceLROnPlateau
from torch.utils.data import DataLoader
from gerbil_train.data.tfrecord_dataset import collect_tfrecord_part_files, count_tfrecord_records
from gerbil_train.metrics.classification import auc
from gerbil_train.utils.nn import count_parameters, print_model_structure
from gerbil_train.utils.seed import set_seed

__all__ = ["BaseTrainer"]


class BaseTrainer:
    """Base trainer abstraction for model training, evaluation, and utilities."""

    def __init__(
        self,
        *,
        model: nn.Module,
        optimizer: Optimizer,
        scheduler: LRScheduler | ReduceLROnPlateau | None,
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
        """
        self.model: nn.Module = model
        self.optimizer: Optimizer = optimizer
        self.scheduler: LRScheduler | ReduceLROnPlateau | None = scheduler
        self.device: torch.device = torch.device(device)
        self.gradient_clip_norm: float | None = gradient_clip_norm
        self.monitor: str = monitor
        self.monitor_mode: str = monitor_mode
        self.patience: int = patience
        self.best_checkpoint_path: Path | None = Path(best_checkpoint_path)
        self.best_metric: float | None = best_metric
        self.wait: int = wait
        self.seed: int | None = seed
        self.verbose: bool = verbose

        self.current_epoch: int = 0
        self.epochs: int = 0
        self.global_step: int = 0
        self.writer = None
        self.batch_inspector = None
        self.total_train_samples: int = 0
        self.steps_per_epoch: int = 0
        self._profile_path: Path | None = None
        self._epoch_start_time: float = 0.0
        self.plot_path: Path | None = None

        self.train_loss_history: list[float] = []
        self.val_loss_history: list[float] = []
        self.model_name: str = "Model"
        self._metric_key: str = "auc"


    def set_batch_inspector(self, inspector: Any) -> None:
        """Attach a batch inspector for logging training batches."""
        self.batch_inspector = inspector


    def setup_total_train_samples(self, data_cfg: dict, batch_size: int, split: str = "train") -> None:
        """Compute total training samples from data config and set on trainer."""
        root_dir = Path(data_cfg["paths"]["tfrecord_root"]) / data_cfg.get("split_subdirs", {}).get(split, split) / "tfrecord"
        train_files = collect_tfrecord_part_files(root_dir)
        self.total_train_samples = count_tfrecord_records(train_files)
        self.steps_per_epoch = (self.total_train_samples + batch_size - 1) // batch_size


    def inspect_batch(self, step: int, batch: Any) -> None:
        """Inspect a training batch if a batch inspector is attached.
        Call this inside ``train_one_epoch`` at the desired step.
        """
        if self.batch_inspector is not None:
            self.batch_inspector(step, batch, self.current_epoch)


    def set_profile_path(self, run_dir: str | Path) -> None:
        """Set the path for saving training profile logs."""
        self._profile_path = Path(run_dir) / "profile.txt"


    def finalize_epoch(self, epoch: int, metrics: dict[str, float], message: str) -> None:
        """Finalize epoch by logging a message with elapsed time and steps per second."""
        elapsed = time.time() - self._epoch_start_time
        steps_per_sec = self.steps_per_epoch / elapsed if elapsed > 0 and self.steps_per_epoch > 0 else 0.0
        message += f" | steps: {self.steps_per_epoch} | steps/s: {steps_per_sec:.2f} | time: {elapsed:.1f}s"
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


    def build_result(self) -> Any:
        """Construct a training result object from the collected training state."""
        raise NotImplementedError


    def fit(self, train_loader: Any, val_loader: Any = None, test_loader: Any = None) -> Any:
        """Run the full training lifecycle with the given dataloaders."""
        self.on_fit_start(train_loader, val_loader, test_loader)
        self.fit_epochs()
        return self.build_result()
    

    def fit_epochs(self) -> None:
        """Run the full training lifecycle."""
        self.setup()
        self.on_train_start()

        if self.total_train_samples > 0:
            print(f"Train samples: {self.total_train_samples} | Steps/epoch: {self.steps_per_epoch}")

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
            set_seed(self.seed)

        self.model.to(self.device)

        if self.verbose:
            print_model_structure(self.model)
            count_parameters(self.model)

        if self.best_checkpoint_path is not None:
            self.best_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)


    def cleanup(self) -> None:
        """Release resources after training completes.

        Subclasses can override this to close writers, files, or other handles.
        """
        pass


    def on_train_start(self) -> None:
        """Hook called before the first training epoch."""
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
        """ Hook called after validation ends."""
        v = metrics.get(self._metric_key)
        if v is not None:
            self.scheduler_step(v)


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
        """ Run evaluation on the given dataloader and return metrics."""
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


    def scheduler_step(self, metric: float | None = None) -> None:
        """Advance the learning rate scheduler.

        :param metric: Metric required by ``ReduceLROnPlateau``
        """
        if self.scheduler is None:
            return
        if isinstance(self.scheduler, ReduceLROnPlateau):
            if metric is None:
                raise ValueError("metric is required for ReduceLROnPlateau")
            self.scheduler.step(metric) # loss 不降时衰减 LR
            return
        # StepLR, CosineAnnealingLR, etc. do not require a metric
        self.scheduler.step() # 按固定 epoch 步长衰减 LR


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
                print(f"Saved best model to {self.best_checkpoint_path.resolve()}")
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
        """Emit a trainer log message."""
        print(message)
