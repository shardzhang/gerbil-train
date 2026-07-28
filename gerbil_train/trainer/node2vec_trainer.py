"""Trainer for Node2Vec: graph construction + biased walks + Skip-gram."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F

from gerbil_train.config.train_config import TrainConfig
from gerbil_train.trainer.base_trainer import BaseTrainer
from gerbil_train.utils.embedding import bag_to_padded

__all__ = ["Node2VecTrainer"]


class Node2VecTrainer(BaseTrainer):
    """Builds co-occurrence graph, generates biased walks, trains Skip-gram."""

    def __init__(self, model: nn.Module, train_cfg: TrainConfig, data_cfg: dict[str, Any] | None = None) -> None:
        optimizer_cfg = train_cfg.optimizer
        checkpoint_cfg = train_cfg.checkpoint
        early_stop_cfg = train_cfg.early_stop
        logging_cfg = train_cfg.logging

        lr = optimizer_cfg.lr
        wd = optimizer_cfg.weight_decay
        self.num_walks = 50
        self.walk_length = 20
        self.p = 1.0
        self.q = 0.5
        self.window_size = 5
        self.num_neg = 5

        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
        super().__init__(
            model=model,
            optimizer=optimizer,
            scheduler=None,
            device=train_cfg.device or ("cuda" if torch.cuda.is_available() else "cpu"),
            gradient_clip_norm=None,
            monitor="val_loss",
            monitor_mode="min",
            patience=0 if not early_stop_cfg.enabled else int(early_stop_cfg.patience),
            best_checkpoint_path=checkpoint_cfg.path,
            best_metric=None,
            wait=0,
            seed=train_cfg.seed,
            verbose=logging_cfg.verbose,
        )

        self.model_name = "Node2Vec"
        self.epochs = int(train_cfg.epochs)
        self.vocab_size = int(model.vocab_size)
        self.item_field = model.item_field

        # Build graph and precompute walks (done once, can be regenerated)
        self.walk_sequences: list[list[int]] = []
        self.walker = None

    def build_graph_and_walks(self, dataloader) -> None:
        """Extract sequences from data, build graph, generate walks."""
        from gerbil_train.models.node2vec import build_cooccurrence_graph, _BiasedRandomWalker

        device = next(self.model.parameters()).device

        # Extract all sequences from the dataset
        sequences: list[list[int]] = []
        for batch in dataloader:
            batch = self.move_batch_to_device(batch)
            feature_bags = batch["feature_bags"]
            padded_ids, padded_weights, lengths, max_seq_len = bag_to_padded(
                feature_bags[self.item_field], device)
            for s in range(padded_ids.size(0)):
                L = int(lengths[s].item())
                if L >= 2:
                    sequences.append(padded_ids[s, :L].tolist())

        if not sequences:
            raise ValueError("No valid sequences found in data")

        # Build weighted adjacency matrix
        adj = build_cooccurrence_graph(sequences, self.vocab_size, self.window_size)
        print(f"Graph built: {self.vocab_size} nodes, {(adj > 0).sum()} edges")

        # Create biased random walker
        self.walker = _BiasedRandomWalker(adj, p=self.p, q=self.q, walk_length=self.walk_length)

        # Generate walks
        rng = np.random.default_rng(42)
        walk_seqs: list[list[int]] = []
        start_nodes = list(range(self.vocab_size))
        for _ in range(self.num_walks):
            rng.shuffle(start_nodes)
            for v in start_nodes:
                w = self.walker.walk(v, rng)
                if len(w) >= 2:
                    walk_seqs.append(w)
        self.walk_sequences = walk_seqs
        print(f"Generated {len(walk_seqs)} walks")

    def forward_step(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        """Skip-gram on precomputed walk sequences."""
        device = next(self.model.parameters()).device
        model = self.model
        ws = self.window_size
        nneg = self.num_neg
        vocab = self.vocab_size

        # Use precomputed walks (batch is ignored for generation;
        # in practice walks can be regenerated every N epochs)
        if not self.walk_sequences:
            return {"loss": torch.tensor(0.0, device=device, requires_grad=True)}

        all_targets, all_contexts, all_labels = [], [], []
        for seq in self.walk_sequences:
            L = len(seq)
            for t in range(L):
                start = max(0, t - ws)
                end = min(L, t + ws + 1)
                ctx = seq[start:t] + seq[t + 1:end]
                n_pos = len(ctx)
                if n_pos == 0:
                    continue
                all_targets.extend([seq[t]] * n_pos)
                all_contexts.extend(ctx)
                all_labels.extend([1.0] * n_pos)

                all_targets.extend([seq[t]] * (n_pos * nneg))
                all_contexts.extend(np.random.randint(0, vocab, n_pos * nneg).tolist())
                all_labels.extend([0.0] * (n_pos * nneg))

        tgt = torch.tensor(all_targets, device=device)
        ctx = torch.tensor(all_contexts, device=device)
        lbl = torch.tensor(all_labels, device=device).float()
        logits = model(tgt, ctx)
        loss = F.binary_cross_entropy_with_logits(logits, lbl)
        return {"loss": loss}

    def compute_loss(self, outputs, batch):
        return outputs["loss"]

    def compute_metrics(self, outputs, batch):
        return {"loss": float(outputs["loss"].item())}

    def on_train_start(self):
        super().on_train_start()
        # Build graph and walks from the training dataloader
        self.build_graph_and_walks(self.train_dataloader)

    def fit(self, train_loader, val_loader=None, test_loader=None):
        self.train_dataloader = train_loader
        super().fit(epochs=self.epochs, train_dataloader=train_loader, val_dataloader=val_loader, test_dataloader=test_loader)

    @torch.no_grad()
    def evaluate(self, dataloader=None):
        return {}
