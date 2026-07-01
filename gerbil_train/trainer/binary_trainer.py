"""Shared trainer for binary classification models (GwEN Binary, DIN, DeepFM, etc)."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from tqdm.auto import tqdm

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from gerbil_train.config.train_config import TrainConfig
from gerbil_train.trainer.base_trainer import BaseTrainer
from gerbil_train.utils import BatchInspector
from gerbil_train.utils.plot import save_curve_values
from gerbil_train.metrics.classification import auc, average_precision, gauc, map_score, mrr_score


class BinaryClassificationTrainer(BaseTrainer):
    """Shared trainer for binary classification models (GwEN Binary, DIN, DeepFM, etc).

    Provides the full training lifecycle: train/validate/evaluate with BCE loss,
    per-user metrics (GAUC, MAP, MRR), checkpointing, and training curve plotting.
    Subclasses only need to set ``model_name`` and optionally override
    ``forward_step`` / ``compute_loss``.
    """
    def __init__(self, model: nn.Module, train_cfg: TrainConfig, data_cfg: dict[str, Any] | None = None) -> None:
        """Initialize the binary classification trainer.

        :param model: PyTorch model to train (should output logits or probabilities)
        :param train_cfg: Training configuration (optimizer, scheduler, checkpointing, etc)
        :param data_cfg: fixme. Optional data configuration (used to compute total training samples) 
        """
        optimizer_cfg = train_cfg.optimizer
        scheduler_cfg = train_cfg.scheduler
        checkpoint_cfg = train_cfg.checkpoint
        early_stop_cfg = train_cfg.early_stop
        logging_cfg = train_cfg.logging

        optimizer = self._create_optimizer(model, optimizer_cfg)
        self._initial_lr = float(optimizer.param_groups[0].get("lr", optimizer_cfg.lr))
        self._scheduler_cfg = scheduler_cfg

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=str(scheduler_cfg.mode),
            factor=float(scheduler_cfg.factor),
            patience=int(scheduler_cfg.patience),
        ) if scheduler_cfg.enabled else None

        super().__init__(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            device=train_cfg.device or ("cuda" if torch.cuda.is_available() else "cpu"),
            gradient_clip_norm=None,
            monitor=str(checkpoint_cfg.monitor or "val_gauc"),
            monitor_mode=str(checkpoint_cfg.mode or "max"),
            patience=0 if not early_stop_cfg.enabled else int(early_stop_cfg.patience),
            best_checkpoint_path=checkpoint_cfg.path,
            best_metric=None,
            wait=0,
            seed=train_cfg.seed,
            verbose=logging_cfg.verbose,
        )

        self.model_name = "BinaryModel"
        self.config = train_cfg
        self.epochs = int(train_cfg.epochs)

        self.train_loader: DataLoader | None = None
        self.validation_loader: DataLoader | None = None
        self.test_loader: DataLoader | None = None

        self.train_loss_history: list[float] = []
        self.val_loss_history: list[float] = []
        self.val_auc_history: list[float] = []
        self.val_ap_history: list[float] = []
        self.val_gauc_history: list[float] = []
        self.val_map_history: list[float] = []
        self.val_mrr_history: list[float] = []
        self.plot_path = Path(logging_cfg.plot_path) if logging_cfg.plot_path is not None else None

        if data_cfg is not None:
            self.setup_total_train_samples(data_cfg, train_cfg.data.batch_size)

        if train_cfg.inspector.enabled:
            self.set_batch_inspector(BatchInspector(
                log_first=train_cfg.inspector.log_first,
                log_every=train_cfg.inspector.log_every,
            ))


    def _create_optimizer(self, model: nn.Module, optimizer_cfg: Any) -> optim.Optimizer:
        """Create optimizer. Override in subclasses (e.g. FTRL) for different optimizers."""
        return torch.optim.Adam(
            model.parameters(),
            lr=float(optimizer_cfg.lr or 1e-3),
            weight_decay=float(optimizer_cfg.weight_decay or 0.0),
        )


    def fit(self, train_loader: DataLoader, validation_loader: DataLoader | None, test_loader: DataLoader | None = None) -> None:
        """Run the full training lifecycle with the given data loaders."""
        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.test_loader = test_loader

        self.train_loss_history.clear()
        self.val_loss_history.clear()

        self.val_auc_history.clear()
        self.val_gauc_history.clear()
        self.val_ap_history.clear()
        self.val_map_history.clear()
        self.val_mrr_history.clear()

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
            model_out = self.forward_step(batch)
            loss = self.compute_total_loss(model_out, batch)
            self.zero_grad()              # clean the gradient
            self.backward_step(loss)      # compute the gradient
            self.clip_gradients()         # clip the gradients
            self.optimizer_step()         # update the parameters
            total_loss += float(loss.item())
            total_steps += 1
            train_pbar.set_postfix(loss=f"{total_loss / step:.4f}")
            self.global_step += 1
            self.update_learning_rate(self.global_step)
        avg_loss = total_loss / max(total_steps, 1)
        return {"loss": avg_loss}


    def forward_step(self, batch: dict[str, Any]) -> torch.Tensor:
        """Forward pass: return sigmoid probabilities."""
        return self.model(batch["feature_bags"])


    def compute_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute binary cross-entropy loss."""
        import torch.nn.functional as F
        return F.binary_cross_entropy(logits, targets)


    def compute_total_loss(self, outputs: torch.Tensor, batch: dict[str, Any]) -> torch.Tensor:
        """Compute total loss. Override in subclasses (e.g. DIEN) for auxiliary losses."""
        return self.compute_loss(outputs, batch["targets"].float())


    def on_epoch_end(self, epoch: int, metrics: dict[str, float]) -> None:
        """Log metrics and build message for finalize_epoch."""
        train_loss = metrics.get("train_loss")
        val_loss = metrics.get("val_loss")
        val_auc = metrics.get("val_auc")
        val_ap = metrics.get("val_ap")
        val_gauc = metrics.get("val_gauc")
        val_map = metrics.get("val_map")
        val_mrr = metrics.get("val_mrr")

        if train_loss is not None:
            self.train_loss_history.append(float(train_loss))
        if val_loss is not None:
            self.val_loss_history.append(float(val_loss))
        if val_auc is not None:
            self.val_auc_history.append(float(val_auc))
        if val_gauc is not None:
            self.val_gauc_history.append(float(val_gauc))
        if val_ap is not None:
            self.val_ap_history.append(float(val_ap))
        if val_map is not None:
            self.val_map_history.append(float(val_map))
        if val_mrr is not None:
            self.val_mrr_history.append(float(val_mrr))

        message = f"Epoch {epoch + 1} | loss: {train_loss:.4f}" if train_loss is not None else f"Epoch {epoch + 1}"
        if val_loss is not None:
            message += f" | val_loss: {val_loss:.4f}"
        if val_auc is not None:
            message += f" | auc: {val_auc:.4f}"
        if val_gauc is not None:
            message += f" | gauc: {val_gauc:.4f}"
        if val_ap is not None:
            message += f" | ap: {val_ap:.4f}"
        if val_map is not None:
            message += f" | map: {val_map:.4f}"
        if val_mrr is not None:
            message += f" | mrr: {val_mrr:.4f}"
        self.finalize_epoch(epoch, metrics, message)


    @torch.no_grad()
    def validate(self, epoch: int | None = None) -> dict[str, float]:
        """Evaluate on validation set: AUC, AP, GAUC, MAP, MRR."""
        if self.validation_loader is None:
            return {}

        self.model.eval()
        all_uids: list[torch.Tensor] = []
        all_labels: list[torch.Tensor] = []
        all_scores: list[torch.Tensor] = []
        total_loss = 0.0
        total_steps = 0
        for batch in self.validation_loader:
            batch = self.move_batch_to_device(batch)
            sigmoids = self.forward_step(batch)
            targets = batch["targets"].float()
            total_loss += self.compute_loss(sigmoids, targets).item()
            uid_bag = batch["feature_bags"].get("user_id")
            if uid_bag is None:
                raise ValueError("GAUC requires 'user_id' in feature_bags")
            offsets = uid_bag["offsets"]
            ends = torch.cat([offsets[1:], offsets.new_tensor([len(uid_bag["indices"])])])
            all_uids.append(uid_bag["indices"][ends - 1])
            all_labels.append(targets)
            all_scores.append(sigmoids)
            total_steps += 1

        cat_labels: torch.Tensor = torch.cat(all_labels)
        cat_scores: torch.Tensor = torch.cat(all_scores)
        result = {
            "loss": round(total_loss / max(total_steps, 1), 4),
            "auc": round(auc(cat_labels, cat_scores), 4),
            "ap": round(average_precision(cat_labels, cat_scores), 4),
        }
        if all_uids:
            cat_uids: torch.Tensor = torch.cat(all_uids)
            valid_mask = cat_uids != 0
            if valid_mask.any():
                valid_uids = cat_uids[valid_mask]
                valid_labels = cat_labels[valid_mask]
                valid_scores = cat_scores[valid_mask]
                result["gauc"] = round(gauc(valid_uids, valid_labels, valid_scores), 4)
                result["map"] = round(map_score(valid_uids, valid_labels, valid_scores, weighted=True), 4)
                result["mrr"] = round(mrr_score(valid_uids, valid_labels, valid_scores, weighted=True), 4)
        return result


    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader | None = None) -> dict[str, float]:
        """Evaluate on test set: AUC, AP, GAUC, MAP, MRR."""
        if dataloader is None:
            return {}

        self.model.eval()
        all_uids: list[torch.Tensor] = []
        all_labels: list[torch.Tensor] = []
        all_scores: list[torch.Tensor] = []
        total_steps = 0
        for batch in dataloader:
            batch = self.move_batch_to_device(batch)
            sigmoids = self.forward_step(batch)
            targets = batch["targets"].float()
            uid_bag = batch["feature_bags"].get("user_id")
            if uid_bag is None:
                raise ValueError("GAUC requires 'user_id' in feature_bags")
            offsets = uid_bag["offsets"]
            ends = torch.cat([offsets[1:], offsets.new_tensor([len(uid_bag["indices"])])])
            all_uids.append(uid_bag["indices"][ends - 1])
            all_labels.append(targets)
            all_scores.append(sigmoids)
            total_steps += 1

        cat_labels: torch.Tensor = torch.cat(all_labels)
        cat_scores: torch.Tensor = torch.cat(all_scores)
        result = {
            "test_auc": round(auc(cat_labels, cat_scores), 4),
            "test_ap": round(average_precision(cat_labels, cat_scores), 4),
        }
        if all_uids:
            cat_uids: torch.Tensor = torch.cat(all_uids)
            valid_mask = cat_uids != 0
            if valid_mask.any():
                valid_uids = cat_uids[valid_mask]
                valid_labels = cat_labels[valid_mask]
                valid_scores = cat_scores[valid_mask]
                result["test_gauc"] = round(gauc(valid_uids, valid_labels, valid_scores), 4)
                result["test_map"] = round(map_score(valid_uids, valid_labels, valid_scores, weighted=True), 4)
                result["test_mrr"] = round(mrr_score(valid_uids, valid_labels, valid_scores, weighted=True), 4)
        return result


    def save_training_artifacts(self) -> None:
        """Save loss and metric curves after training completes."""
        if self.plot_path is None:
            return
        save_curve_values(self.train_loss_history, self.plot_path.with_name(f"{self.plot_path.stem}_loss.txt"))
        save_curve_values(self.val_auc_history, self.plot_path.with_name(f"{self.plot_path.stem}_metric.txt"))
        self.plot_loss_curve()
        self.plot_metric_curve()


    def plot_loss_curve(self) -> None:
        """Plot training and validation loss curves."""
        if self.plot_path is None or not self.train_loss_history:
            return
        from matplotlib import pyplot as plt
        plt.figure(figsize=(8, 4))
        epochs = range(1, len(self.train_loss_history) + 1)
        plt.plot(epochs, self.train_loss_history, label="train_loss")
        if self.val_loss_history:
            plt.plot(epochs, self.val_loss_history, label="val_loss")
        plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title(f"{self.model_name} Training & Validation Loss")
        plt.grid(True, linestyle="--", alpha=0.3); plt.legend(); plt.tight_layout()
        self.plot_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(self.plot_path.with_name(f"{self.plot_path.stem}_loss.png"))
        plt.close()


    def plot_metric_curve(self) -> None:
        """ Plot validation metric curve (AUC) after training completes.
        fixme: add gauc
        """
        if self.plot_path is None or not self.val_auc_history:
            return
        from matplotlib import pyplot as plt
        plt.figure(figsize=(8, 4))
        epochs = range(1, len(self.val_auc_history) + 1)
        plt.plot(epochs, self.val_auc_history, label="auc")
        plt.xlabel("Epoch"); plt.ylabel("auc"); plt.title(f"{self.model_name} Validation auc")
        plt.grid(True, linestyle="--", alpha=0.3); plt.legend(); plt.tight_layout()
        self.plot_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(self.plot_path.with_name(f"{self.plot_path.stem}_metric.png"))
        plt.close()
