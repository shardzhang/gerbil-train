"""Base trainer abstractions and lifecycle hooks."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler, ReduceLROnPlateau

from gerbil_train.metrics.classification import auc
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
        self.best_checkpoint_path: Path | None = (
            Path(best_checkpoint_path) if best_checkpoint_path is not None else None
        )
        self.best_metric: float | None = best_metric
        self.wait: int = wait
        self.seed: int | None = seed

        self.current_epoch: int = 0
        self.max_epochs: int = 0
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
        self.val_metric_history: list[float] = []
        self.model_name: str = "Model"
        self.metric_name: str = "Metric"
        self._metric_key: str = "auc"
        self._compute_metric = staticmethod(auc)

    def _on_fit_start(self, train_loader: Any, val_loader: Any, test_loader: Any) -> None:
        self.train_loader = train_loader
        self.validation_loader = val_loader
        self.test_loader = test_loader
        self.train_loss_history.clear()
        self.val_metric_history.clear()

    def _build_result(self) -> Any:
        return None

    def fit(self, train_loader: Any, val_loader: Any = None, test_loader: Any = None) -> Any:
        self._on_fit_start(train_loader, val_loader, test_loader)
        self.max_epochs = self.epochs
        self._fit_epochs()
        return self._build_result()

    def set_batch_inspector(self, inspector: Any) -> None:
        self.batch_inspector = inspector

    def set_total_train_samples(self, total: int, batch_size: int) -> None:
        """Set total training samples and compute steps per epoch."""
        self.total_train_samples = total
        self.steps_per_epoch = (total + batch_size - 1) // batch_size

    def inspect_batch(self, step: int, batch: Any) -> None:
        """Inspect a training batch if a batch inspector is attached.

        Call this inside ``train_one_epoch`` at the desired step.
        """
        if self.batch_inspector is not None:
            self.batch_inspector(step, batch, self.current_epoch)

    def set_profile_path(self, run_dir: str | Path) -> None:
        self._profile_path = Path(run_dir) / "profile.txt"

    def finalize_epoch(self, epoch: int, metrics: dict[str, float], message: str) -> None:
        elapsed = time.time() - self._epoch_start_time
        steps_per_sec = self.steps_per_epoch / elapsed if elapsed > 0 and self.steps_per_epoch > 0 else 0.0
        message += f" | steps: {self.steps_per_epoch} | steps/s: {steps_per_sec:.2f} | time: {elapsed:.1f}s"
        self.log_message(message)
        if self._profile_path is not None:
            self._profile_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._profile_path, "a") as f:
                f.write(message + "\n")

    def setup(self) -> None:
        """Prepare trainer state before training starts.

        The default implementation applies the configured random seed if present.
        Subclasses can extend this to create loggers, writers, directories, or caches.
        """
        if self.seed is not None:
            set_seed(self.seed)

        self.model.to(self.device)

        if self.best_checkpoint_path is not None:
            self.best_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    def cleanup(self) -> None:
        """Release resources after training completes.

        Subclasses can override this to close writers, files, or other handles.
        """

    def _fit_epochs(self) -> None:
        """Run the full training lifecycle."""
        self.setup()
        self.on_train_start()

        if self.total_train_samples > 0:
            print(f"Train samples: {self.total_train_samples} | Steps/epoch: {self.steps_per_epoch}")

        try:
            for epoch in range(epochs):
                self.current_epoch = epoch
                self.on_epoch_start(epoch)

                train_metrics = self.train_one_epoch(epoch)
                metrics = {
                    f"train_{key}": value for key, value in train_metrics.items()
                }

                val_metrics = self.validate(epoch)
                metrics.update(
                    {f"val_{key}": value for key, value in val_metrics.items()}
                )

                self.on_validation_end(metrics)
                self.on_epoch_end(epoch, metrics)

                if self.update_best_state(metrics):
                    break
        finally:
            self.on_train_end()
            self.cleanup()

    """
    2. on-train-start hook
    """

    def on_train_start(self) -> None:
        """Hook called before the first training epoch."""

    """
    4. on-train-end hook
    """

    def on_train_end(self) -> None:
        """Hook called once after the training loop ends.

        The default implementation persists training artifacts such as saved
        curve values and plots.
        """
        self.save_training_artifacts()

    """
    3.1 on-epoch-start hook
    """

    def on_epoch_start(self, epoch: int) -> None:
        """Hook called before each epoch starts.

        :param epoch: Zero-based epoch index
        """
        self._epoch_start_time = time.time()

    def train_one_epoch(self, epoch: int) -> dict[str, float]:
        if self.train_loader is None:
            return {"loss": 0.0}
        self.model.train()
        total_loss = 0.0
        from tqdm.auto import tqdm
        pbar = tqdm(self.train_loader, total=self.steps_per_epoch or None, desc=f"Epoch {epoch + 1}/{self.max_epochs} [train]", leave=False)
        for step, batch in enumerate(pbar, start=1):
            batch = self.move_batch_to_device(batch)
            self.on_train_step_start(batch)
            self.inspect_batch(step, batch)
            self.zero_grad()
            outputs = self.forward_step(batch)
            loss = self.compute_loss(outputs, batch)
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
        total_metric = 0.0
        total_loss = 0.0
        total_steps = 0
        import torch.nn.functional as F
        with torch.no_grad():
            for batch in self.validation_loader:
                batch = self.move_batch_to_device(batch)
                outputs = self.forward_step(batch)
                labels = batch["targets"].float()
                total_loss += float(F.binary_cross_entropy(outputs, labels).item())
                total_metric += self._compute_metric(labels, outputs)
                total_steps += 1
        d = max(total_steps, 1)
        return {"loss": total_loss / d, self._metric_key: total_metric / d}

    def evaluate(self, dataloader: DataLoader | None = None) -> dict[str, float]:
        if dataloader is None:
            return {}
        self.model.eval()
        total_metric = 0.0
        total_steps = 0
        with torch.no_grad():
            for batch in dataloader:
                batch = self.move_batch_to_device(batch)
                outputs = self.forward_step(batch)
                labels = batch["targets"].float()
                total_metric += self._compute_metric(labels, outputs)
                total_steps += 1
        return {f"test_{self._metric_key}": total_metric / max(total_steps, 1)}

    def on_validation_end(self, metrics: dict[str, float]) -> None:
        v = metrics.get(self._metric_key)
        if v is not None:
            self.scheduler_step(v)

    def on_epoch_end(self, epoch: int, metrics: dict[str, float]) -> None:
        train_loss = metrics.get("train_loss")
        val_metric = metrics.get(f"val_{self._metric_key}")
        if train_loss is not None:
            self.train_loss_history.append(float(train_loss))
        if val_metric is not None:
            self.val_metric_history.append(float(val_metric))
        msg = f"Epoch {epoch + 1} | loss: {train_loss:.4f}" if train_loss is not None else f"Epoch {epoch + 1}"
        if val_metric is not None:
            msg += f" | {self.metric_name} val: {val_metric:.4f}"
        self.finalize_epoch(epoch, metrics, msg)

    def on_train_step_start(self, batch: Any) -> None:
        pass

    def on_train_step_end(self, metrics: dict[str, float]) -> None:
        pass

    def zero_grad(self) -> None:
        self.optimizer.zero_grad(set_to_none=True)

    def forward_step(self, batch: Any) -> Any:
        return self.model(batch["feature_bags"])

    def compute_loss(self, outputs: torch.Tensor, batch: Any) -> torch.Tensor:
        import torch.nn.functional as F
        return F.binary_cross_entropy(outputs, batch["targets"].float())

    """
    compute metrics
    """

    def compute_metrics(self, outputs: torch.Tensor, batch: Any) -> dict[str, float]:
        """Compute metrics from model outputs and label tensors."""
        raise NotImplementedError

    """
    backward step
    """

    def backward_step(self, loss: torch.Tensor) -> None:
        """Run backpropagation for one loss value."""
        loss.backward()

    """
    clip gradients
    """

    def clip_gradients(self) -> None:
        """Clip gradients when a max norm is configured."""
        if self.gradient_clip_norm is None:
            return
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            max_norm=self.gradient_clip_norm,
        )

    """
    optimizer step
    """

    def optimizer_step(self) -> None:
        """Apply one optimizer update."""
        self.optimizer.step()

    """
    scheduler step
    """

    def scheduler_step(self, metric: float | None = None) -> None:
        """Advance the learning rate scheduler.

        :param metric: Metric required by ``ReduceLROnPlateau``
        """
        if self.scheduler is None:
            return
        if isinstance(self.scheduler, ReduceLROnPlateau):
            if metric is None:
                raise ValueError("metric is required for ReduceLROnPlateau")
            self.scheduler.step(metric)
            return
        self.scheduler.step()

    """
    3.3 validate
    """

    def validate(self, epoch: int | None = None) -> dict[str, float]:
        """Run validation.

        :param epoch: Optional zero-based epoch index
        :return: Validation metrics
        """
        # self.on_validation_start(epoch)
        # val_metrics = self.validate(epoch)
        # self.on_validation_end(val_metrics)
        raise NotImplementedError

    def on_validation_start(self, epoch: int) -> None:
        """Hook called before validation starts.

        This hook is called once per epoch before the validation routine runs,
        even if the concrete ``validate`` implementation later returns an empty
        metrics dictionary.

        :param epoch: Zero-based epoch index
        """

    def on_validation_end(self, metrics: dict[str, float]) -> None:
        """Hook called after validation ends.

        :param metrics: Validation metrics for the current epoch
        """

    """
    evaluate
    """

    def evaluate(self, *args: Any, **kwargs: Any) -> dict[str, float]:
        """Run evaluation outside the training loop."""
        # self.on_evaluate_start hook
        # self.forward_step
        # self.on_evaluate_end hook
        raise NotImplementedError

    def on_evaluate_start(self) -> None:
        """Hook called before evaluation starts."""

    def on_evaluate_end(self, metrics: dict[str, float]) -> None:
        """Hook called after evaluation ends.

        :param metrics: Evaluation metrics
        """

    """
    predict
    """

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
                key: self.move_batch_to_device(value) for key, value in batch.items()
            }

        if isinstance(batch, list):
            return [self.move_batch_to_device(value) for value in batch]

        if isinstance(batch, tuple):
            return tuple(self.move_batch_to_device(value) for value in batch)

        return batch

    def log_message(self, message: str) -> None:
        """Emit a trainer log message."""
        print(message)

    def save_training_artifacts(self) -> None:
        """Persist all training artifacts collected by the trainer."""
        self.save_loss_curve()
        self.save_metric_curve()
        self.plot_loss_curve()
        self.plot_metric_curve()

    def save_loss_curve(self) -> None:
        if self.plot_path is None or not self.train_loss_history:
            return
        from gerbil_train.utils.plot import save_curve_values
        save_curve_values(self.train_loss_history, self.plot_path.with_name(f"{self.plot_path.stem}_loss.txt"))

    def save_metric_curve(self) -> None:
        if self.plot_path is None or not self.val_metric_history:
            return
        from gerbil_train.utils.plot import save_curve_values
        save_curve_values(self.val_metric_history, self.plot_path.with_name(f"{self.plot_path.stem}_metric.txt"))

    def plot_loss_curve(self) -> None:
        if self.plot_path is None or not self.train_loss_history:
            return
        from matplotlib import pyplot as plt
        plt.figure(figsize=(8, 4))
        plt.plot(range(1, len(self.train_loss_history) + 1), self.train_loss_history, label="train_loss")
        plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title(f"{self.model_name} Training Loss")
        plt.grid(True, linestyle="--", alpha=0.3); plt.legend(); plt.tight_layout()
        self.plot_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(self.plot_path.with_name(f"{self.plot_path.stem}_loss.png"))
        plt.close()

    def plot_metric_curve(self) -> None:
        if self.plot_path is None or not self.val_metric_history:
            return
        from matplotlib import pyplot as plt
        plt.figure(figsize=(8, 4))
        plt.plot(range(1, len(self.val_metric_history) + 1), self.val_metric_history, label=f"val_{self.metric_name}")
        plt.xlabel("Epoch"); plt.ylabel(self.metric_name); plt.title(f"{self.model_name} Validation {self.metric_name}")
        plt.grid(True, linestyle="--", alpha=0.3); plt.legend(); plt.tight_layout()
        self.plot_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(self.plot_path.with_name(f"{self.plot_path.stem}_metric.png"))
        plt.close()
