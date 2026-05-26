from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.optim as optim
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from gerbil_train.losses.ranking import compute_loss
from gerbil_train.metrics.ranking import ndcg_score
from gerbil_train.trainer.base_trainer import BaseTrainer
from gerbil_train.utils.plot import (
    load_curve_values,
    save_curve_values,
    save_training_curves,
)

__all__ = [
    "LearningToRankTrainer",
    "LearningToRankTrainingResult",
]


@dataclass
class LearningToRankTrainingResult:
    """Container for aggregated learning-to-rank training results."""

    train_loss_history: list[float]
    val_ndcg_history: list[float]
    best_ndcg: float


class LearningToRankTrainer(BaseTrainer):
    """Trainer for pointwise, pairwise, and listwise learning-to-rank models."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        scheduler: Any = None,
        device: torch.device | str = "cpu",
        gradient_clip_norm: float | None = None,
        monitor: str = "val_ndcg",
        monitor_mode: str = "max",
        patience: int = 5,
        best_checkpoint_path: str | Path | None = None,
        best_metric: float | None = None,
        wait: int = 0,
        seed: int | None = 42,
        log_dir: str | Path | None = None,
        plot_path: str | Path | None = None,
        val_k: int = 5,
    ) -> None:
        """Initialize the learning-to-rank trainer.

        :param model: Ranking model that produces one score per document
        :param optimizer: Optimizer used to update model parameters
        :param scheduler: Optional learning rate scheduler
        :param device: Device used for training and evaluation
        :param gradient_clip_norm: Optional gradient clipping threshold
        :param monitor: Metric name used for checkpointing and early stopping
        :param monitor_mode: Whether smaller or larger monitored values are better
        :param patience: Early stopping patience measured in validation checks
        :param best_checkpoint_path: Optional destination path for the best checkpoint
        :param best_metric: Initial best monitored metric
        :param wait: Initial early stopping wait counter
        :param seed: Optional random seed for reproducibility
        :param log_dir: Optional TensorBoard log directory
        :param plot_path: Optional output path for the rendered training-curves figure
        :param val_k: Cutoff used for validation NDCG@k
        """
        super().__init__(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            gradient_clip_norm=gradient_clip_norm,
            monitor=monitor,
            monitor_mode=monitor_mode,
            patience=patience,
            best_checkpoint_path=best_checkpoint_path,
            best_metric=best_metric,
            wait=wait,
            seed=seed,
        )

        self.val_k = val_k
        self.log_dir = Path(log_dir) if log_dir is not None else None
        self.plot_path = Path(plot_path) if plot_path is not None else None
        self.train_loss_history: list[float] = []
        self.val_ndcg_history: list[float] = []
        self.train_loader: DataLoader | None = None
        self.val_loader: DataLoader | None = None
        self.loss_name = ""

    def setup(self) -> None:
        """Prepare training directories and tensorboard writer.

        The default setup path may initialize the random seed, create output
        directories, and open the TensorBoard writer.
        """
        super().setup()

        if self.log_dir is not None:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            self.writer = SummaryWriter(log_dir=str(self.log_dir))

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        *,
        loss_name: str,
        epochs: int,
    ) -> LearningToRankTrainingResult:
        """Run learning-to-rank training and return summarized history.

        :param train_loader: Training dataloader
        :param val_loader: Validation dataloader
        :param loss_name: Ranking loss name to optimize
        :param epochs: Number of training epochs
        :return: Aggregated training history and best score
        """
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_name = loss_name
        self.train_loss_history.clear()
        self.val_ndcg_history.clear()

        super().fit(epochs=epochs)

        return LearningToRankTrainingResult(
            train_loss_history=list(self.train_loss_history),
            val_ndcg_history=list(self.val_ndcg_history),
            best_ndcg=self.best_metric or 0.0,
        )

    def cleanup(self) -> None:
        """Close runtime resources such as the TensorBoard writer."""
        if self.writer is not None:
            self.writer.close()
            self.writer = None

    def _curve_text_paths(self) -> tuple[Path | None, Path | None]:
        """Return the text-file paths used for persisted training curves."""
        if self.plot_path is None:
            return None, None
        loss_path = self.plot_path.with_name(f"{self.plot_path.stem}_loss.txt")
        metric_path = self.plot_path.with_name(f"{self.plot_path.stem}_metric.txt")
        return loss_path, metric_path

    def save_loss_curve(self) -> None:
        """Save training-loss values to a text file."""
        loss_path, _ = self._curve_text_paths()
        if loss_path is None:
            return
        save_curve_values(self.train_loss_history, loss_path)

    def save_metric_curve(self) -> None:
        """Save validation-metric values to a text file."""
        _, metric_path = self._curve_text_paths()
        if metric_path is None:
            return
        save_curve_values(self.val_ndcg_history, metric_path)

    def plot_loss_curve(self) -> None:
        """Render the combined training-curves figure from saved values."""
        if self.plot_path is None:
            return

        loss_path, metric_path = self._curve_text_paths()
        train_loss_history = (
            load_curve_values(loss_path)
            if loss_path is not None and loss_path.exists()
            else self.train_loss_history
        )
        val_ndcg_history = (
            load_curve_values(metric_path)
            if metric_path is not None and metric_path.exists()
            else self.val_ndcg_history
        )
        save_training_curves(
            train_loss_history,
            val_ndcg_history,
            self.plot_path,
        )

    def plot_metric_curve(self) -> None:
        """Render the metric-related figure artifact."""
        self.plot_loss_curve()

    def on_validation_end(self, metrics: dict[str, float]) -> None:
        """Advance the scheduler after validation.

        :param metrics: Validation metrics for the current epoch
        """
        ndcg = metrics.get("ndcg")
        if ndcg is not None:
            self.scheduler_step(ndcg)

    def train_one_epoch(self, epoch: int) -> dict[str, float]:
        """Train over all ranking groups for one epoch.

        :param epoch: Zero-based epoch index
        :return: Epoch-level training metrics
        """
        self.model.train()
        epoch_display = epoch + 1

        total_loss = 0.0
        train_pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch_display}/{self.max_epochs} [train]",
            leave=False,
        )
        for step, batch in enumerate(train_pbar, start=1):
            step_metrics = self.train_one_step(batch)
            total_loss += step_metrics["loss"]
            train_pbar.set_postfix(loss=f"{total_loss / step:.4f}")
        return {"loss": total_loss / len(self.train_loader)}

    def on_epoch_end(self, epoch: int, metrics: dict[str, float]) -> None:
        """Record epoch history, log TensorBoard scalars, and print progress.

        The ``metrics`` dictionary already contains the aggregated ``train_*``
        and ``val_*`` keys for the current epoch, which makes this hook the
        natural place to update history buffers, write TensorBoard summaries,
        and print epoch-level logs before early stopping is checked.

        :param epoch: Zero-based epoch index
        :param metrics: Aggregated train/validation metrics for the epoch
        """
        train_loss = metrics.get("train_loss")
        val_ndcg = metrics.get("val_ndcg")

        if train_loss is not None:
            self.train_loss_history.append(train_loss)

        if val_ndcg is not None:
            self.val_ndcg_history.append(val_ndcg)

        if self.writer is not None:
            if train_loss is not None:
                self.writer.add_scalar("Loss/train", train_loss, epoch)

            if val_ndcg is not None:
                self.writer.add_scalar("NDCG/val", val_ndcg, epoch)

        if train_loss is not None and val_ndcg is not None:
            self.log_message(
                f"Epoch {epoch + 1:2d} | loss: {train_loss:.4f} | NDCG@{self.val_k} val: {val_ndcg:.4f}"
            )

    def train_one_step(self, batch: dict[str, Any]) -> dict[str, float]:
        """Run one optimization step on a single query group.

        :param batch: Query-group batch with feature and label tensors
        :return: Step-level training metrics
        """
        self.on_train_step_start(batch)

        batch = self.move_batch_to_device(batch)
        self.zero_grad()
        outputs = self.forward_step(batch)
        loss = self.compute_loss(batch, outputs)
        self.backward_step(loss)
        self.clip_gradients()
        self.optimizer_step()
        self.global_step += 1
        metrics = {"loss": float(loss.item())}

        self.on_train_step_end(metrics)
        return metrics

    def validate(self, epoch: int | None = None) -> dict[str, float]:
        """Evaluate NDCG on the validation split.

        :param epoch: Optional zero-based epoch index
        :return: Validation metrics dictionary
        """
        self.model.eval()
        epoch_index = self.current_epoch if epoch is None else epoch
        epoch_display = epoch_index + 1
        ndcg_sum = 0.0

        with torch.no_grad():
            val_pbar = tqdm(
                self.val_loader,
                desc=f"Epoch {epoch_display}/{self.max_epochs} [val]",
                leave=False,
            )
            for step, batch in enumerate(val_pbar, start=1):
                batch = self.move_batch_to_device(batch)
                outputs = self.forward_step(batch)
                ndcg_sum += self.compute_metrics(batch, outputs)["ndcg"]
                val_pbar.set_postfix(ndcg=f"{ndcg_sum / step:.4f}")
        return {"ndcg": ndcg_sum / len(self.val_loader)}

    def evaluate(
        self,
        dataloader: DataLoader | None = None,
        ks: Sequence[int] = (5,),
    ) -> dict[int, float]:
        """Evaluate the model on a split using one or more NDCG cutoffs.

        :param dataloader: Optional dataloader to evaluate; defaults to validation loader
        :param ks: Sequence of NDCG cutoffs to compute
        :return: Mapping from cutoff k to mean NDCG@k
        """
        self.on_evaluate_start()
        dataloader = self.val_loader if dataloader is None else dataloader
        self.model.eval()
        ndcg_totals = {k: 0.0 for k in ks}

        with torch.no_grad():
            for batch in dataloader:
                batch = self.move_batch_to_device(batch)
                outputs = self.forward_step(batch)
                for k in ks:
                    ndcg_totals[k] += float(ndcg_score(batch["y"], outputs, k=k))

        metrics = {k: ndcg_totals[k] / len(dataloader) for k in ks}
        self.on_evaluate_end({f"ndcg@{k}": value for k, value in metrics.items()})
        return metrics

    def forward_step(self, batch: dict[str, Any]):
        """Run the ranking model on one query group's feature matrix.

        :param batch: Query-group batch containing the ``X`` feature tensor
        :return: Predicted document scores for the group
        """
        return self.model(batch["X"])

    def compute_loss(self, batch: dict[str, Any], outputs: Any) -> torch.Tensor:
        """Compute the configured ranking loss for one query group.

        :param batch: Query-group batch containing labels in ``y``
        :param outputs: Predicted document scores
        :return: Scalar loss tensor
        """
        return compute_loss(self.loss_name, outputs, batch["y"])

    def compute_metrics(self, batch: dict[str, Any], outputs: Any) -> dict[str, float]:
        """Compute validation NDCG for one query group.

        :param batch: Query-group batch containing labels in ``y``
        :param outputs: Predicted document scores
        :return: Metric dictionary for the query group
        """
        return {"ndcg": float(ndcg_score(batch["y"], outputs, k=self.val_k))}
