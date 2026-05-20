from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn, optim
from torch.utils.data import DataLoader
from gerbil_train.models.shared_bottom_two_tower import SharedBottomTwoTower

@dataclass
class StageResult:
    loss: float
    steps: int


class SharedBottomTwoTowerTrainer:
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
    ) -> None:
        self.model = model.to(device)
        self.device = device

        self.implicit_optimizer = optim.Adam(
            self.model.parameters(),
            lr=implicit_lr,
            weight_decay=weight_decay,
        )
        self.explicit_optimizer = optim.Adam(
            self.model.parameters(),
            lr=explicit_lr,
            weight_decay=weight_decay,
        )

    def fit(self, implicit_loader: DataLoader | None, explicit_loader: DataLoader | None, implicit_epochs: int = 1, explicit_epochs: int = 1) -> None:
        """Run the two-stage training pipeline.
        :param implicit_loader: DataLoader for the implicit pre-training stage (can be None to skip)
        :param explicit_loader: DataLoader for the explicit fine-tuning stage (can be None to skip)
        :param implicit_epochs: Number of epochs to train on the implicit task (default: 1)
        :param explicit_epochs: Number of epochs to train on the explicit task (default: 1)
        """
        if implicit_loader is not None:
            for epoch in range(implicit_epochs):
                result = self.train_implicit_one_epoch(implicit_loader)
                print(
                    f"[implicit] epoch={epoch + 1} "
                    f"loss={result.loss:.6f} steps={result.steps}"
                )

        if explicit_loader is not None:
            for epoch in range(explicit_epochs):
                result = self.train_explicit_one_epoch(explicit_loader)
                print(
                    f"[explicit] epoch={epoch + 1} "
                    f"loss={result.loss:.6f} steps={result.steps}"
                )

    def train_implicit_one_epoch(self, dataloader: DataLoader) -> StageResult:
        """Train one epoch on the implicit task."""
        self.model.train()

        total_loss = 0.0
        total_steps = 0

        for batch in dataloader:
            batch = self._move_batch_to_device(batch)

            query_features = batch["query_features"]              # [B, Dq]
            pos_item_features = batch["pos_item_features"]        # [B, Di]
            neg_item_features = batch["neg_item_features"]        # [B, N, Di]

            loss = self.compute_implicit_loss(
                query_features=query_features,
                pos_item_features=pos_item_features,
                neg_item_features=neg_item_features,
            )

            self.implicit_optimizer.zero_grad()
            loss.backward()
            self.implicit_optimizer.step()

            total_loss += float(loss.item())
            total_steps += 1

        avg_loss = total_loss / max(total_steps, 1)
        return StageResult(loss=avg_loss, steps=total_steps)

    def train_explicit_one_epoch(self, dataloader: DataLoader) -> StageResult:
        """Train one epoch on the explicit task."""
        self.model.train()

        total_loss = 0.0
        total_steps = 0

        for batch in dataloader:
            batch = self._move_batch_to_device(batch)

            query_features = batch["query_features"]      # [B, Dq]
            item_features = batch["item_features"]        # [B, Di]
            labels = batch["label"].float()               # [B]

            loss = self.compute_explicit_loss(
                query_features=query_features,
                item_features=item_features,
                labels=labels,
            )

            self.explicit_optimizer.zero_grad()
            loss.backward()
            self.explicit_optimizer.step()

            total_loss += float(loss.item())
            total_steps += 1

        avg_loss = total_loss / max(total_steps, 1)
        return StageResult(loss=avg_loss, steps=total_steps)

    def compute_implicit_loss(self, query_features: Tensor, pos_item_features: Tensor, neg_item_features: Tensor) -> Tensor:
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
        neg_item_features_flat = neg_item_features.view(batch_size * num_negatives, item_dim)

        # [B*N, E] -> [B, N, E]
        neg_i_emb = self.model.encode_item_implicit(neg_item_features_flat).view(batch_size, num_negatives, -1)

        # negative scores: [B, N]
        neg_scores = torch.sum(q_emb.unsqueeze(1) * neg_i_emb, dim=-1)
        # neg_scores = torch.bmm(neg_i_emb, q_emb.unsqueeze(-1)).squeeze(-1)  # [B, N]

        # logits: positive at index 0
        # [B, 1+N]
        logits = torch.cat([pos_score, neg_scores], dim=1)

        # labels: positive item is class 0
        targets = torch.zeros(batch_size, dtype=torch.long, device=logits.device)

        return F.cross_entropy(logits, targets)

    def compute_explicit_loss(self, query_features: Tensor, item_features: Tensor, labels: Tensor) -> Tensor:
        """Compute explicit task loss with stop-gradient on shared-bottom.
        :param query_features: [batch_size, query_input_dim]
        :param item_features: [batch_size, item_input_dim]
        :param labels: [batch_size] with relevance scores
        :return: Tensor representing the loss
        """
        outputs = self.model(
            query_features=query_features,
            item_features=item_features,
            detach_shared_for_explicit=True,
        )
        scores = outputs.explicit_score
        return F.mse_loss(scores, labels)

    def evaluate_explicit(self, dataloader: DataLoader) -> dict[str, float]:
        """Evaluate explicit task."""
        self.model.eval()

        total_loss = 0.0
        total_steps = 0

        with torch.no_grad():
            for batch in dataloader:
                batch = self._move_batch_to_device(batch)

                query_features = batch["query_features"]
                item_features = batch["item_features"]
                labels = batch["label"].float()

                loss = self.compute_explicit_loss(
                    query_features=query_features,
                    item_features=item_features,
                    labels=labels,
                )

                total_loss += float(loss.item())
                total_steps += 1

        avg_loss = total_loss / max(total_steps, 1)
        return {"mse": avg_loss}

    def _move_batch_to_device(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Move batch to the specified device.
        :param batch: Dictionary containing batch data
        :return: Dictionary with batch data moved to the specified device
        """
        output: dict[str, Any] = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                output[key] = value.to(self.device)
            else:
                output[key] = value
        return output