from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, optim
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from gerbil_train.models.shared_bottom_two_tower import SharedBottomTwoTower
from gerbil_train.trainer.base_trainer import BaseTrainer

__all__ = ["SharedBottomTwoTowerTrainer", "StageResult"]


@dataclass
class StageResult:
    """Container for averaged stage loss and processed step count."""

    loss: float
    steps: int


class SharedBottomTwoTowerTrainer(BaseTrainer):
    """Trainer for the Shared-Bottom Two-Tower (SBTT) model.

    This trainer implements the two-stage training pipeline proposed in
    "Improving Relevance Prediction with Transfer Learning in Large-scale Retrieval Systems"

    Training Stages:
        1. Pre-train on implicit user behavior data
        2. Fine-tune on explicit relevance labeled data
    """

    def __init__(
        self,
        model: SharedBottomTwoTower,
        device: str = "cuda",
        implicit_lr: float = 1e-3,
        explicit_lr: float = 1e-4,
        weight_decay: float = 1e-5,
        gradient_clip_norm: float | None = None,
        monitor: str = "train_loss",
        monitor_mode: str = "min",
        patience: int = 0,
        best_checkpoint_path: str | Path | None = None,
        best_metric: float | None = None,
        wait: int = 0,
        seed: int | None = 42,
    ) -> None:
        """Initialize the shared-bottom two-tower trainer.

        :param model: Shared-bottom two-tower model instance
        :param device: Device used for training and evaluation
        :param implicit_lr: Learning rate for the implicit stage optimizer
        :param explicit_lr: Learning rate for the explicit stage optimizer
        :param weight_decay: Weight decay applied to both optimizers
        :param gradient_clip_norm: Optional gradient clipping threshold
        :param monitor: Metric name used for checkpointing and early stopping
        :param monitor_mode: Whether smaller or larger monitored values are better
        :param patience: Early stopping patience measured in epochs
        :param best_checkpoint_path: Optional destination path for the best checkpoint
        :param best_metric: Initial best monitored metric
        :param wait: Initial early stopping wait counter
        :param seed: Optional random seed for reproducibility
        """
        explicit_optimizer = optim.Adam(
            model.parameters(),
            lr=explicit_lr,
            weight_decay=weight_decay,
        )
        super().__init__(
            model=model,
            optimizer=explicit_optimizer,
            scheduler=None,
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

        self.implicit_optimizer = optim.Adam(
            self.model.parameters(),
            lr=implicit_lr,
            weight_decay=weight_decay,
        )
        self.explicit_optimizer = self.optimizer
        self.current_stage = "explicit"
        self.current_loader: DataLoader | None = None
        self.stage_epochs = 0

    def fit(
        self,
        implicit_loader: DataLoader | None,
        explicit_loader: DataLoader | None,
        implicit_epochs: int = 1,
        explicit_epochs: int = 1,
    ) -> None:
        """Run the two-stage training pipeline.

        :param implicit_loader: Dataloader for implicit training; can be ``None`` to skip
        :param explicit_loader: Dataloader for explicit training; can be ``None`` to skip
        :param implicit_epochs: Number of implicit training epochs
        :param explicit_epochs: Number of explicit training epochs
        """
        self.setup()
        self.on_train_start()
        try:
            if implicit_loader is not None:
                self.current_stage = "implicit"
                self.current_loader = implicit_loader
                self.optimizer = self.implicit_optimizer
                self.stage_epochs = implicit_epochs
                for epoch in range(implicit_epochs):
                    self.current_epoch = epoch
                    self.on_epoch_start(epoch)
                    train_metrics = self.train_one_epoch(epoch)
                    metrics = {
                        f"train_{key}": value for key, value in train_metrics.items()
                    }
                    self.on_epoch_end(epoch, metrics)

            if explicit_loader is not None:
                self.current_stage = "explicit"
                self.current_loader = explicit_loader
                self.optimizer = self.explicit_optimizer
                self.stage_epochs = explicit_epochs
                for epoch in range(explicit_epochs):
                    self.current_epoch = epoch
                    self.on_epoch_start(epoch)
                    train_metrics = self.train_one_epoch(epoch)
                    metrics = {
                        f"train_{key}": value for key, value in train_metrics.items()
                    }
                    self.on_epoch_end(epoch, metrics)
        finally:
            self.on_train_end()
            self.cleanup()

    def on_epoch_end(self, epoch: int, metrics: dict[str, float]) -> None:
        """Log aggregated stage metrics at the end of an epoch.

        :param epoch: Zero-based epoch index within the current stage
        :param metrics: Aggregated stage metrics
        """
        loss = metrics.get("train_loss")
        steps = metrics.get("train_steps")
        if loss is None or steps is None:
            return
        self.log_message(
            f"[{self.current_stage}] epoch={epoch + 1} loss={loss:.6f} steps={int(steps)}"
        )

    def train_one_epoch(self, epoch: int) -> dict[str, float]:
        """Train one epoch for the current stage.

        :param epoch: Zero-based epoch index
        :return: Epoch-level stage metrics
        """
        if self.current_loader is None:
            return {"loss": 0.0, "steps": 0.0}

        self.model.train()
        total_loss = 0.0
        total_steps = 0
        epoch_display = epoch + 1

        train_pbar = tqdm(
            self.current_loader,
            desc=f"[{self.current_stage}] Epoch {epoch_display}/{self.stage_epochs} [train]",
            leave=False,
        )
        for step, batch in enumerate(train_pbar, start=1):
            step_metrics = self.train_one_step(batch)
            total_loss += step_metrics["loss"]
            total_steps += 1
            train_pbar.set_postfix(loss=f"{total_loss / step:.4f}")

        avg_loss = total_loss / max(total_steps, 1)
        return {"loss": avg_loss, "steps": float(total_steps)}

    def train_one_step(self, batch: dict[str, Any]) -> dict[str, float]:
        """Run one optimization step for the current stage.

        :param batch: Training batch for either implicit or explicit stage
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
        """Run validation.

        Validation is not part of the current SBTT training flow, so this
        method returns an empty metric dictionary.

        :param epoch: Optional zero-based epoch index
        :return: Empty validation metrics dictionary
        """
        if epoch is not None:
            self.current_epoch = epoch
        return {}


    def evaluate(self) -> dict[str, float]:
        """Run evaluation.

        The generic evaluation interface is not used in the current SBTT flow.

        :return: Empty evaluation metrics dictionary
        """
        self.on_evaluate_start()
        metrics: dict[str, float] = {}
        self.on_evaluate_end(metrics)
        return metrics

    def forward_step(self, batch: dict[str, Any]):
        """Prepare model inputs for the current stage.

        :param batch: Stage-specific batch dictionary
        :return: Intermediate implicit inputs or explicit model outputs
        """
        if self.current_stage == "implicit":
            return {
                "query_features": batch["query_features"],
                "pos_item_features": batch["pos_item_features"],
                "neg_item_features": batch["neg_item_features"],
            }

        outputs = self.model(
            query_features=batch["query_features"],
            item_features=batch["item_features"],
            detach_shared_for_explicit=True,
        )
        return outputs

    def compute_loss(self, batch: dict[str, Any], outputs: Any) -> torch.Tensor:
        """Dispatch to the implicit or explicit loss computation.

        :param batch: Stage-specific batch dictionary
        :param outputs: Outputs returned by ``forward_step``
        :return: Scalar loss tensor
        """
        if self.current_stage == "implicit":
            return self.compute_implicit_loss(
                query_features=outputs["query_features"],
                pos_item_features=outputs["pos_item_features"],
                neg_item_features=outputs["neg_item_features"],
            )

        return self.compute_explicit_loss(batch, outputs)

    def compute_metrics(
        self, _batch: dict[str, Any], _outputs: Any
    ) -> dict[str, float]:
        """Compute stage metrics.

        The current trainer only reports loss, so no additional metrics are returned.

        :param _batch: Stage-specific batch dictionary
        :param _outputs: Outputs returned by ``forward_step``
        :return: Empty metrics dictionary
        """
        return {}

    def save_checkpoint(self, path: str) -> None:
        """Save model and both optimizer states to a checkpoint file.

        :param path: Destination checkpoint path
        """
        checkpoint_path = Path(path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "implicit_optimizer_state_dict": self.implicit_optimizer.state_dict(),
            "explicit_optimizer_state_dict": self.explicit_optimizer.state_dict(),
            "config": self.config,
            "best_metric": self.best_metric,
            "current_stage": self.current_stage,
        }
        torch.save(checkpoint, checkpoint_path)

    def load_checkpoint(self, path: str) -> None:
        """Load model and optimizer states from a checkpoint file.

        :param path: Source checkpoint path
        """
        checkpoint = torch.load(path, map_location=self.device)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])
            if "implicit_optimizer_state_dict" in checkpoint:
                self.implicit_optimizer.load_state_dict(
                    checkpoint["implicit_optimizer_state_dict"]
                )
            if "explicit_optimizer_state_dict" in checkpoint:
                self.explicit_optimizer.load_state_dict(
                    checkpoint["explicit_optimizer_state_dict"]
                )
            self.best_metric = checkpoint.get("best_metric", self.best_metric)
            self.current_stage = checkpoint.get("current_stage", self.current_stage)
            self.optimizer = (
                self.implicit_optimizer
                if self.current_stage == "implicit"
                else self.explicit_optimizer
            )
            return

        self.model.load_state_dict(checkpoint)

    def compute_implicit_loss(
        self,
        query_features: Tensor,
        pos_item_features: Tensor,
        neg_item_features: Tensor,
    ) -> Tensor:
        """Compute implicit task loss.

        Uses positive item + sampled negative items.

        :param query_features: [batch_size, query_input_dim]
        :param pos_item_features: [batch_size, item_input_dim]
        :param neg_item_features: [batch_size, num_negatives, item_input_dim]
        :return: Tensor representing the loss
        """
        # [B, E]
        q_emb = self.model.encode_query_implicit(query_features)
        # [B, E]
        pos_i_emb = self.model.encode_item_implicit(pos_item_features)

        # positive score: [B, 1]
        pos_score = torch.sum(q_emb * pos_i_emb, dim=-1, keepdim=True)
        # pos_score = torch.einsum("bd,bd->b", q_emb, pos_i_emb).unsqueeze(1)  # [B, 1]

        # flatten negative items: [B, N, Di] -> [B*N, Di]
        batch_size, num_negatives, item_dim = neg_item_features.shape
        neg_item_features_flat = neg_item_features.view(
            batch_size * num_negatives, item_dim
        )

        # [B*N, E] -> [B, N, E]
        neg_i_emb = self.model.encode_item_implicit(neg_item_features_flat).view(
            batch_size, num_negatives, -1
        )

        # negative scores: [B, N]
        neg_scores = torch.sum(q_emb.unsqueeze(1) * neg_i_emb, dim=-1)
        # neg_scores = torch.bmm(neg_i_emb, q_emb.unsqueeze(-1)).squeeze(-1)  # [B, N]

        # logits: positive at index 0
        # [B, 1+N]
        logits = torch.cat([pos_score, neg_scores], dim=1)

        # labels: positive item is class 0
        targets = torch.zeros(batch_size, dtype=torch.long, device=logits.device)

        return F.cross_entropy(logits, targets)

    def compute_explicit_loss(self, batch: dict[str, Any], outputs: Any) -> Tensor:
        """Compute explicit task loss with stop-gradient on shared-bottom.

        :param batch: Batch dictionary containing features and labels
        :param outputs: Model outputs containing explicit score
        :return: Tensor representing the loss
        """
        labels = batch["label"].float()
        scores = outputs.explicit_score
        return F.mse_loss(scores, labels)

    def evaluate_explicit(self, dataloader: DataLoader) -> dict[str, float]:
        """Evaluate the explicit stage using mean squared error.

        :param dataloader: Dataloader for explicit evaluation
        :return: Dictionary containing explicit-stage MSE
        """
        self.model.eval()

        total_loss = 0.0
        total_steps = 0
        previous_stage = self.current_stage

        with torch.no_grad():
            self.current_stage = "explicit"
            for batch in dataloader:
                batch = self.move_batch_to_device(batch)
                outputs = self.forward_step(batch)
                loss = self.compute_explicit_loss(batch, outputs)

                total_loss += float(loss.item())
                total_steps += 1

        self.current_stage = previous_stage
        avg_loss = total_loss / max(total_steps, 1)
        return {"mse": avg_loss}
