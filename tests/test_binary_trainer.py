"""Tests for binary_trainer.py — BCE training, validation, metrics."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

import torch
import torch.nn as nn

from gerbil_train.config.train_config import TrainConfig
from gerbil_train.trainer.binary_trainer import BinaryClassificationTrainer


class TinyBinaryModel(nn.Module):
    """Minimal model that returns sigmoid probabilities."""
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 1)

    def forward(self, feature_bags: dict) -> torch.Tensor:
        return torch.sigmoid(self.fc(torch.randn(2, 4))).squeeze(-1)


class BinaryClassificationTrainerTests(unittest.TestCase):
    """Tests for BinaryClassificationTrainer core methods."""

    def test_compute_loss_bce(self) -> None:
        """BCE loss with near-perfect predictions is near zero."""
        logits = torch.tensor([0.999, 0.001, 0.999, 0.001])
        targets = torch.tensor([1.0, 0.0, 1.0, 0.0])
        trainer = self._make_trainer([0] * 10)
        loss = trainer.compute_loss(logits, targets)
        self.assertAlmostEqual(loss.item(), 0.0, places=2)

    def test_compute_loss_bce_wrong(self) -> None:
        """BCE loss with wrong preds is positive."""
        logits = torch.tensor([0.1, 0.9])
        targets = torch.tensor([1.0, 0.0])
        trainer = self._make_trainer([0] * 10)
        loss = trainer.compute_loss(logits, targets)
        self.assertGreater(loss.item(), 1.0)

    def test_compute_total_loss_default(self) -> None:
        """compute_total_loss delegates to compute_loss by default."""
        logits = torch.tensor([0.9, 0.2])
        batch = {"targets": torch.tensor([1, 0])}
        trainer = self._make_trainer([0] * 10)
        total = trainer.compute_total_loss(logits, batch)
        direct = trainer.compute_loss(logits, batch["targets"].float())
        self.assertAlmostEqual(total.item(), direct.item(), places=6)

    def test_forward_step(self) -> None:
        """forward_step returns model output of shape [batch_size]."""
        model = TinyBinaryModel()
        cfg = self._make_minimal_config()
        trainer = BinaryClassificationTrainer(model, cfg)
        batch = {"feature_bags": {}, "targets": torch.tensor([0, 1])}
        out = trainer.forward_step(batch)
        self.assertEqual(len(out.shape), 1)  # 1D output
        self.assertGreater(out.shape[0], 0)

    def test_train_one_epoch_returns_loss(self) -> None:
        """train_one_epoch returns a finite loss value."""
        model = TinyBinaryModel()
        cfg = self._make_minimal_config()
        trainer = BinaryClassificationTrainer(model, cfg)
        loader = self._make_dummy_loader()
        trainer.train_loader = loader
        result = trainer.train_one_epoch(epoch=0)
        self.assertIn("loss", result)
        self.assertGreater(result["loss"], 0.0)
        self.assertLess(result["loss"], 1.0)

    def _make_trainer(self, data: list) -> BinaryClassificationTrainer:
        model = TinyBinaryModel()
        cfg = self._make_minimal_config()
        return BinaryClassificationTrainer(model, cfg)

    def _make_minimal_config(self) -> TrainConfig:
        return TrainConfig.from_dict({
            "seed": 42,
            "device": "cpu",
            "epochs": 1,
            "checkpoint": {"monitor": "val_gauc", "mode": "max"},
            "data": {"batch_size": 2},
            "optimizer": {"lr": 0.001},
        })

    def _make_dummy_loader(self) -> MagicMock:
        loader = MagicMock()
        loader.__iter__.return_value = [
            {"feature_bags": {}, "targets": torch.tensor([0, 1])},
            {"feature_bags": {}, "targets": torch.tensor([1, 0])},
        ]
        loader.__len__.return_value = 2
        return loader


if __name__ == "__main__":
    unittest.main()
