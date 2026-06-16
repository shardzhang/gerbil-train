from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.optim as optim
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from matplotlib import pyplot as plt

from gerbil_train.config import DeepFMTrainConfig
from gerbil_train.losses.ranking import compute_loss as compute_ranking_loss
from gerbil_train.metrics.ranking import ndcg_score
from gerbil_train.trainer.base_trainer import BaseTrainer
from gerbil_train.utils.plot import save_curve_values


__all__ = ["LearningToRankTrainer", "LearningToRankTrainingResult"]


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
        config: DeepFMTrainConfig,
        *,
        loss_name: str = "lambdarank",
    ) -> None:
        """Initialize the learning-to-rank trainer.

        :param model: Ranking model that produces one score per document
        :param config: Training configuration
        :param loss_name: Ranking loss type (lambdarank, listnet, etc.)
        """
        optimizer_cfg = config.optimizer
        scheduler_cfg = config.scheduler
        checkpoint_cfg = config.checkpoint
        early_stop_cfg = config.early_stop
        logging_cfg = config.logging
        evaluation_cfg = config.evaluation

        optimizer = optim.Adam(
            model.parameters(),
            lr=float(optimizer_cfg.lr or 1e-3),
            weight_decay=float(optimizer_cfg.weight_decay or 0.0),
        )

        scheduler = None
        if scheduler_cfg.enabled:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode=str(scheduler_cfg.mode),
                factor=float(scheduler_cfg.factor),
                patience=int(scheduler_cfg.patience),
            )

        device = config.device or ("cuda" if torch.cuda.is_available() else "cpu")

        super().__init__(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            gradient_clip_norm=None,
            monitor=str(checkpoint_cfg.monitor or "val_ndcg"),
            monitor_mode=str(checkpoint_cfg.mode or "max"),
            patience=0 if not early_stop_cfg.enabled else int(early_stop_cfg.patience),
            best_checkpoint_path=checkpoint_cfg.path,
            best_metric=None,
            wait=0,
            seed=config.seed,
        )

        self.config = config
        self.epochs = int(config.epochs)
        self.loss_name = loss_name
        self.topk = int(evaluation_cfg.topk)
        self.train_loader: DataLoader | None = None
        self.val_loader: DataLoader | None = None
        self.train_loss_history: list[float] = []
        self.val_ndcg_history: list[float] = []
        self.log_dir = None if logging_cfg.plot_path is None else Path(logging_cfg.plot_path).parent
        self.plot_path = Path(logging_cfg.plot_path) if logging_cfg.plot_path is not None else None
        self.train_loss_history: list[float] = []
        self.val_ndcg_history: list[float] = []
        self.train_loader: DataLoader | None = None
        self.val_loader: DataLoader | None = None
        self.loss_name = str(config.get("loss_name", "lambdarank"))
        self.epochs = int(config.get("epochs", 30))

    def setup(self) -> None:
        """Prepare training directories and tensorboard writer.

        The default setup path may initialize the random seed, create output
        directories, and open the TensorBoard writer.
        """
        super().setup()

        if self.log_dir is not None:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            self.writer = SummaryWriter(log_dir=str(self.log_dir))

    def cleanup(self) -> None:
        """Close runtime resources such as the TensorBoard writer."""
        if self.writer is not None:
            self.writer.close()
            self.writer = None

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> LearningToRankTrainingResult:
        """Run learning-to-rank training and return summarized history.

        :param train_loader: Training dataloader
        :param val_loader: Validation dataloader
        :return: Aggregated training history and best score
        """
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.train_loss_history.clear()
        self.val_ndcg_history.clear()

        super().fit(epochs=self.epochs)

        return LearningToRankTrainingResult(
            train_loss_history=list(self.train_loss_history),
            val_ndcg_history=list(self.val_ndcg_history),
            best_ndcg=self.best_metric or 0.0,
        )

    def train_one_epoch(self, epoch: int) -> dict[str, float]:
        """Train over all ranking groups for one epoch.

        :param epoch: Zero-based epoch index
        :return: Epoch-level training metrics
        """
        self.model.train()
        epoch_display = epoch + 1

        total_loss = 0.0
        train_pbar = tqdm(
            self.train_loader, total=self.steps_per_epoch or None,
            desc=f"Epoch {epoch_display}/{self.max_epochs} [train]",
            leave=False,
        )
        for step, batch in enumerate(train_pbar, start=1):
            step_metrics = self.train_one_step(batch)
            total_loss += step_metrics["loss"]
            train_pbar.set_postfix(loss=f"{total_loss / step:.4f}")
        return {"loss": total_loss / len(self.train_loader)}

    def train_one_step(self, batch: dict[str, Any]) -> dict[str, float]:
        """Run one optimization step on a single query group.

        :param batch: Query-group batch with feature and label tensors
        :return: Step-level training metrics
        """
        self.on_train_step_start(batch)

        batch = self.move_batch_to_device(batch)
        self.zero_grad()
        outputs = self.forward_step(batch)
        loss = self.compute_loss(outputs, batch)
        self.backward_step(loss)
        self.clip_gradients()
        self.optimizer_step()
        self.global_step += 1
        metrics = {"loss": float(loss.item())}

        self.on_train_step_end(metrics)
        return metrics

    def on_train_end(self):
        self.save_training_curves()

    def validate(self, epoch: int | None = None) -> dict[str, float]:
        """Evaluate NDCG on the validation split.

        :param epoch: Optional zero-based epoch index
        :return: Validation metrics dictionary
        """
        # self.on_validation_start(epoch)

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
                ndcg_sum += self.compute_metrics(outputs, y=batch["y"])["ndcg"]
                val_pbar.set_postfix(ndcg=f"{ndcg_sum / step:.4f}")
        metrics = {"ndcg": ndcg_sum / len(self.val_loader)}

        self.on_validation_end(metrics)
        return metrics

    def on_validation_end(self, metrics: dict[str, float]) -> None:
        """Advance the scheduler after validation.

        :param metrics: Validation metrics for the current epoch
        """
        ndcg = metrics.get("ndcg")
        if ndcg is not None:
            self.scheduler_step(ndcg)

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

        message = ""
        if train_loss is not None and val_ndcg is not None:
            message = f"Epoch {epoch + 1:2d} | loss: {train_loss:.4f} | NDCG@{self.topk} val: {val_ndcg:.4f}"
        if message:
            self.finalize_epoch(epoch, metrics, message)

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
        # self.on_evaluate_start()

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

    def on_evaluate_end(self, metrics: dict[str, float]) -> None:
        """Log evaluation results.

        :param metrics: Evaluation metrics for the current evaluation run
        """
        metric_str = " | ".join(f"{key}: {value:.4f}" for key, value in metrics.items())
        self.log_message(f"Evaluation results | {metric_str}")

    def forward_step(self, batch: dict[str, Any]) -> torch.Tensor:
        """Run the ranking model on one query group's feature matrix.

        :param batch: Query-group batch containing the ``X`` feature tensor
        :return: Predicted document scores for the group
        """
        return self.model(batch["X"])

    def compute_loss(self, outputs: torch.Tensor, batch: dict[str, Any]) -> torch.Tensor:
        """Compute the configured ranking loss for one query group.

        :param outputs: Predicted document scores
        :param batch: Query-group batch containing labels in ``y``
        :return: Scalar loss tensor
        """
        return compute_ranking_loss(self.loss_name, outputs, batch["y"], k=self.topk)

    def compute_metrics(self, outputs: torch.Tensor, batch: dict[str, Any]) -> dict[str, float]:
        """Compute validation NDCG for one query group.

        :param outputs: Predicted document scores
        :param batch: Query-group batch containing labels in ``y``
        :return: Metric dictionary for the query group
        """
        return {"ndcg": float(ndcg_score(batch["y"], outputs, k=self.topk))}

    def _curve_text_paths(self) -> tuple[Path | None, Path | None]:
        """Return the text-file paths used for persisted training curves."""
        if self.plot_path is None:
            return None, None

        loss_path = self.plot_path.with_name(f"{self.plot_path.stem}_loss.txt")
        metric_path = self.plot_path.with_name(f"{self.plot_path.stem}_metric.txt")
        return loss_path, metric_path

    def save_training_curves(self) -> None:
        """Save a figure containing training loss and validation NDCG curves.

        :param train_loss_history: Sequence of training loss values by epoch
        :param val_ndcg_history: Sequence of validation NDCG values by epoch
        :param plot_path: Destination file path for the generated figure
        """
        self.plot_path.parent.mkdir(parents=True, exist_ok=True)

        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(self.train_loss_history)
        plt.title("Train Loss")

        plt.subplot(1, 2, 2)
        plt.plot(self.val_ndcg_history)
        plt.title("Val NDCG@5")

        plt.tight_layout()
        plt.savefig(self.plot_path)
        print(f"Saved training curves to {self.plot_path.resolve()}")
        plt.close()

        loss_path, metric_path = self._curve_text_paths()
        save_curve_values(self.train_loss_history, loss_path)
        save_curve_values(self.val_ndcg_history, metric_path)
