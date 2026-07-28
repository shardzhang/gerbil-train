"""Trainer for Word2Vec (Skip-gram with Negative Sampling)."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn, optim
import torch.nn.functional as F

from gerbil_train.config.train_config import TrainConfig
from gerbil_train.trainer.base_trainer import BaseTrainer

__all__ = ["Word2VecTrainer"]


class Word2VecTrainer(BaseTrainer):
    """Trains Word2Vec via negative sampling on behavior sequences."""

    def __init__(self, model: nn.Module, train_cfg: TrainConfig, data_cfg: dict[str, Any] | None = None) -> None:
        optimizer_cfg = train_cfg.optimizer
        scheduler_cfg = train_cfg.scheduler
        checkpoint_cfg = train_cfg.checkpoint
        early_stop_cfg = train_cfg.early_stop
        logging_cfg = train_cfg.logging

        lr = optimizer_cfg.lr
        wd = optimizer_cfg.weight_decay
        self.window_size = 5
        self.num_neg = 5

        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

        super().__init__(
            model=model,
            optimizer=optimizer,
            scheduler=None,
            device=train_cfg.device or ("cuda" if torch.cuda.is_available() else "cpu"),
            gradient_clip_norm=None,
            monitor=str(checkpoint_cfg.monitor or "val_loss"),
            monitor_mode=str(checkpoint_cfg.mode or "min"),
            patience=0 if not early_stop_cfg.enabled else int(early_stop_cfg.patience),
            best_checkpoint_path=checkpoint_cfg.path,
            best_metric=None,
            wait=0,
            seed=train_cfg.seed,
            verbose=logging_cfg.verbose,
        )

        self.model_name = "Word2Vec"
        self.epochs = int(train_cfg.epochs)
        self.vocab_size = int(model.vocab_size)
        self.item_field = model.item_field

    def forward_step(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        from gerbil_train.utils.embedding import bag_to_padded

        feature_bags = batch["feature_bags"]
        device = next(self.model.parameters()).device
        model = self.model
        vocab = self.vocab_size
        ws = self.window_size
        nneg = self.num_neg

        padded_ids, padded_weights, lengths, max_seq_len = bag_to_padded(feature_bags[model.item_field], device)
        B = padded_ids.size(0)

        all_targets, all_contexts, all_labels = [], [], []

        for s in range(B):
            L = int(lengths[s].item())
            if L < 2:
                continue
            seq = padded_ids[s, :L]

            for t in range(L):
                start = max(0, t - ws)
                end = min(L, t + ws + 1)
                ctx = torch.cat([seq[start:t], seq[t + 1:end]])
                n_pos = ctx.size(0)
                if n_pos == 0:
                    continue

                # Positive
                all_targets.append(seq[t].expand(n_pos))
                all_contexts.append(ctx)
                all_labels.append(torch.ones(n_pos, device=device))

                # Negative
                all_targets.append(seq[t].expand(n_pos * nneg))
                all_contexts.append(torch.randint(0, vocab, (n_pos * nneg,), device=device))
                all_labels.append(torch.zeros(n_pos * nneg, device=device))

        if not all_targets:
            return {"loss": torch.tensor(0.0, device=device, requires_grad=True)}

        tgt = torch.cat(all_targets)
        ctx = torch.cat(all_contexts)
        lbl = torch.cat(all_labels)
        logits = model(tgt, ctx)
        loss = F.binary_cross_entropy_with_logits(logits, lbl)
        return {"loss": loss}

    def compute_loss(self, outputs, batch):
        return outputs["loss"]

    def compute_metrics(self, outputs, batch):
        return {"loss": float(outputs["loss"].item())}

    def fit(self, train_loader, val_loader=None, test_loader=None):
        super().fit(epochs=self.epochs, train_dataloader=train_loader, val_dataloader=val_loader, test_dataloader=test_loader)

    @torch.no_grad()
    def evaluate(self, dataloader=None):
        return {}
