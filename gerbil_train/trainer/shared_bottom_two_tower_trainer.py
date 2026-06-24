from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor, optim
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from matplotlib import pyplot as plt

from gerbil_train.metrics.ranking import ndcg_score
from gerbil_train.models.shared_bottom_two_tower import SharedBottomTwoTower
from gerbil_train.trainer.base_trainer import BaseTrainer
from gerbil_train.utils.plot import save_curve_values

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

    def __init__(self, model: nn.Module, config: dict[str, Any]) -> None:
        """Initialize the shared-bottom two-tower trainer.

        :param model: Shared-bottom two-tower model instance
        :param config: Train configuration mapping
        """
        optimizer_cfg = config.get("optimizer", {})
        trainer_cfg = config.get("trainer", {})
        gradient_cfg = config.get("gradient", {})
        checkpoint_cfg = config.get("checkpoint", {})
        early_stop_cfg = config.get("early_stop", {})
        implicit_optimizer_cfg = optimizer_cfg.get("implicit", {})
        explicit_optimizer_cfg = optimizer_cfg.get("explicit", {})

        checkpoint_dir = checkpoint_cfg.get("dir")
        self.checkpoint_dir = (
            Path(checkpoint_dir) if checkpoint_dir is not None else None
        )
        self.save_best_only = bool(checkpoint_cfg.get("save_best_only", False))
        self.save_last = bool(checkpoint_cfg.get("save_last", False))
        self.save_every_epoch = bool(checkpoint_cfg.get("save_every_epoch", False))

        configured_best_checkpoint_path = checkpoint_cfg.get("best_checkpoint_path")
        if configured_best_checkpoint_path is not None:
            best_checkpoint_path = Path(configured_best_checkpoint_path)
        elif self.checkpoint_dir is not None and self.save_best_only:
            best_checkpoint_path = self.checkpoint_dir / "best_model.pth"
        else:
            best_checkpoint_path = None

        self.last_checkpoint_path = (
            self.checkpoint_dir / "last_model.pth"
            if self.checkpoint_dir is not None and self.save_last
            else None
        )

        configured_device = str(config.get("device", "cuda"))
        if configured_device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        else:
            device = configured_device
        implicit_lr = float(implicit_optimizer_cfg.get("lr", 1e-3))
        explicit_lr = float(explicit_optimizer_cfg.get("lr", 1e-4))
        weight_decay = float(implicit_optimizer_cfg.get("weight_decay", 1e-5))
        gradient_clip_norm = gradient_cfg.get("clip_grad_norm")
        monitor = str(checkpoint_cfg.get("monitor", "train_loss"))
        monitor_mode = str(checkpoint_cfg.get("mode", "min"))
        patience = (
            int(early_stop_cfg.get("patience", 0))
            if early_stop_cfg.get("enabled", False)
            else 0
        )
        best_metric = None
        wait = 0
        seed = config.get("seed", 42)

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

        self.config = config
        self.implicit_optimizer = optim.Adam(
            self.model.parameters(),
            lr=implicit_lr,
            weight_decay=weight_decay,
        )
        self.explicit_optimizer = explicit_optimizer
        self.current_stage = "explicit"
        self.current_loader: DataLoader | None = None
        self.validation_loader: DataLoader | None = None
        self.stage_epochs = 0
        self.implicit_epochs = int(trainer_cfg.get("implicit_epochs", 1))
        self.explicit_epochs = int(trainer_cfg.get("explicit_epochs", 1))
        self.implicit_loss_history: list[float] = []
        self.explicit_loss_history: list[float] = []
        evaluation_cfg = config.get("evaluation", {})
        self.validation_k = int(evaluation_cfg.get("validation_k", 10))
        self.validation_history: list[float] = []

    def setup(self) -> None:
        """Prepare trainer resources before training starts."""
        super().setup()
        if self.checkpoint_dir is not None:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def fit(
        self,
        implicit_loader: DataLoader | None,
        explicit_loader: DataLoader | None,
        validation_loader: DataLoader | None = None,
    ) -> None:
        """Run the two-stage training pipeline.

        :param implicit_loader: Dataloader for implicit training; can be ``None`` to skip
        :param explicit_loader: Dataloader for explicit training; can be ``None`` to skip
        :param validation_loader: Optional ranking validation dataloader
        """
        self.setup()
        self.on_train_start()
        self.validation_loader = validation_loader
        try:
            if implicit_loader is not None:
                self.current_stage = "implicit"
                self.current_loader = implicit_loader
                self.optimizer = self.implicit_optimizer
                self.stage_epochs = self.implicit_epochs
                for epoch in range(self.implicit_epochs):
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
                    self.on_epoch_end(epoch, metrics)
                    if self.update_best_state(metrics):
                        break

            if explicit_loader is not None:
                self.current_stage = "explicit"
                self.current_loader = explicit_loader
                self.optimizer = self.explicit_optimizer
                self.stage_epochs = self.explicit_epochs
                for epoch in range(self.explicit_epochs):
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
                    self.on_epoch_end(epoch, metrics)
                    if self.update_best_state(metrics):
                        break
        finally:
            self.on_train_end()
            self.cleanup()

    def on_train_end(self) -> None:
        """Persist configured checkpoint artifacts when training finishes."""
        self.save_training_artifacts()

        if (
            self.save_best_only
            and self.best_checkpoint_path is not None
            and not self.best_checkpoint_path.exists()
        ):
            self.save_checkpoint(self.best_checkpoint_path)
            self.log_message(
                "Monitored metric was unavailable; saved final state as fallback best checkpoint "
                f"to {self.best_checkpoint_path}"
            )

        if self.last_checkpoint_path is not None:
            self.save_checkpoint(self.last_checkpoint_path)
            self.log_message(f"Saved last checkpoint to {self.last_checkpoint_path}")

    def on_epoch_end(self, epoch: int, metrics: dict[str, float]) -> None:
        """Log aggregated stage metrics at the end of an epoch.

        :param epoch: Zero-based epoch index within the current stage
        :param metrics: Aggregated stage metrics
        """
        loss = metrics.get("train_loss")
        steps = metrics.get("train_steps")
        val_ndcg = metrics.get(f"val_ndcg@{self.validation_k}")
        if loss is None or steps is None:
            return
        message = f"[{self.current_stage}] epoch={epoch + 1} loss={loss:.6f} steps={int(steps)}"
        if val_ndcg is not None:
            message += f" val_ndcg@{self.validation_k}={val_ndcg:.6f}"
            self.validation_history.append(float(val_ndcg))
        self.finalize_epoch(epoch, metrics, message)

        if self.current_stage == "implicit":
            self.implicit_loss_history.append(float(loss))
        else:
            self.explicit_loss_history.append(float(loss))

        if self.save_every_epoch and self.checkpoint_dir is not None:
            epoch_checkpoint_path = (
                self.checkpoint_dir / f"{self.current_stage}_epoch_{epoch + 1}.pth"
            )
            self.save_checkpoint(epoch_checkpoint_path)
            self.log_message(f"Saved epoch checkpoint to {epoch_checkpoint_path}")

    def _curve_paths(self) -> tuple[Path | None, Path | None, Path | None]:
        """Return output paths for shared-bottom training artifacts."""
        if self.checkpoint_dir is None:
            return None, None, None
        implicit_loss_path = self.checkpoint_dir / "implicit_loss.txt"
        explicit_loss_path = self.checkpoint_dir / "explicit_loss.txt"
        loss_plot_path = self.checkpoint_dir / "training_loss.png"
        return implicit_loss_path, explicit_loss_path, loss_plot_path

    def save_loss_curve(self) -> None:
        """Persist shared-bottom stage loss histories to text files."""
        pass

    def save_metric_curve(self) -> None:
        """Persist shared-bottom validation metric history to a text file."""
        pass

    def plot_loss_curve(self) -> None:
        pass

    
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
            # 1. self.on_train_step_start hook
            # 2. zero_grad
            # 3. forward_step
            # 4. compute_loss
            # 5. backward_step

            # 6. clip_gradients
            # 7. optimizer_step
            # 8. scheduler_step
            # 9. self.on_train_step_end hook

        :param batch: Training batch for either implicit or explicit stage
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

    def validate(self, epoch: int | None = None) -> dict[str, float]:
        """Run validation.

        :param epoch: Optional zero-based epoch index
        :return: Ranking validation metrics dictionary
        """
        if epoch is not None:
            self.current_epoch = epoch
        if self.validation_loader is None:
            return {}

        self.model.eval()
        ndcg_total = 0.0
        total_steps = 0

        with torch.no_grad():
            for batch in self.validation_loader:
                batch = self.move_batch_to_device(batch)
                outputs = self.model(
                    query_features=batch["query_features"],
                    item_features=batch["item_features"],
                    detach_shared_for_explicit=True,
                )
                ndcg_total += float(
                    ndcg_score(
                        batch["labels"],
                        outputs.explicit_score,
                        k=self.validation_k,
                    )
                )
                total_steps += 1

        return {f"ndcg@{self.validation_k}": ndcg_total / max(total_steps, 1)}

    def evaluate(self, dataloader: DataLoader | None = None) -> dict[str, float]:
        """Run explicit ranking evaluation on a held-out test loader.

        :param dataloader: Test ranking dataloader
        :return: Test ranking metrics dictionary
        """
        self.on_evaluate_start()
        if dataloader is None:
            metrics: dict[str, float] = {}
            self.on_evaluate_end(metrics)
            return metrics

        self.model.eval()
        ndcg_total = 0.0
        total_steps = 0

        with torch.no_grad():
            for batch in dataloader:
                batch = self.move_batch_to_device(batch)
                outputs = self.model(
                    query_features=batch["query_features"],
                    item_features=batch["item_features"],
                    detach_shared_for_explicit=True,
                )
                ndcg_total += float(
                    ndcg_score(
                        batch["labels"],
                        outputs.explicit_score,
                        k=self.validation_k,
                    )
                )
                total_steps += 1

        metrics = {f"test_ndcg@{self.validation_k}": ndcg_total / max(total_steps, 1)}
        self.on_evaluate_end(metrics)
        return metrics

    def forward_step(self, batch: dict[str, Any]) -> dict[str, Any]:
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

    def compute_loss(self, outputs: Any, batch: dict[str, Any]) -> torch.Tensor:
        """Dispatch to the implicit or explicit loss computation.

        :param outputs: Outputs returned by ``forward_step``
        :param batch: Stage-specific batch dictionary
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
        self, outputs: Any, batch: dict[str, Any]
    ) -> dict[str, float]:
        """Compute stage metrics.

        The current trainer only reports loss, so no additional metrics are returned.

        :param outputs: Outputs returned by ``forward_step``
        :param batch: Unused
        :return: Empty metrics dictionary
        """
        return {}

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
