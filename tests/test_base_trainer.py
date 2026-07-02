"""Tests for base_trainer.py — lifecycle, checkpointing, LR scheduling."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import torch
import torch.nn as nn

from gerbil_train.trainer.base_trainer import BaseTrainer


class DummyModel(nn.Module):
    """Minimal model for trainer tests."""
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.fc(x))


class ConcreteTrainer(BaseTrainer):
    """Concrete subclass to test BaseTrainer hooks."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.train_loss_history: list[float] = []
        self._epochs_completed = 0

    def train_one_epoch(self, epoch: int) -> dict[str, float]:
        self._epochs_completed += 1
        return {"loss": 0.5 - epoch * 0.01}

    def validate(self, epoch: int | None = None) -> dict[str, float]:
        return {"val_loss": 0.4 - (epoch or 0) * 0.01}

    def on_epoch_end(self, epoch: int, metrics: dict[str, float]) -> None:
        if metrics.get("train_loss") is not None:
            self.train_loss_history.append(float(metrics["train_loss"]))


class BaseTrainerTests(unittest.TestCase):
    """Tests for BaseTrainer lifecycle and core methods."""

    def _make_trainer(self, monitor: str = "val_loss", monitor_mode: str = "min") -> ConcreteTrainer:
        model = DummyModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        return ConcreteTrainer(
            model=model,
            optimizer=optimizer,
            scheduler=None,
            device="cpu",
            gradient_clip_norm=None,
            monitor=monitor,
            monitor_mode=monitor_mode,
            patience=5,
            best_checkpoint_path=None,
            best_metric=None,
            wait=0,
            seed=42,
        )

    def test_initial_state(self) -> None:
        """Trainer starts with default state values."""
        t = self._make_trainer()
        self.assertEqual(t.current_epoch, 0)
        self.assertEqual(t.global_step, 0)
        self.assertEqual(t.best_metric, None)
        self.assertEqual(t.wait, 0)

    def test_scheduler_step_warmup_exp_decay(self) -> None:
        """scheduler_step applies exponential decay correctly."""
        t = self._make_trainer()
        t._initial_lr = 0.01
        t._scheduler_cfg = type("Cfg", (), {
            "type": "warmup_exp_decay", "warmup_steps": 100,
            "decay_rate": -0.5, "learning_rate_min": 0.0,
            "total_steps": 0,
        })()

        t.scheduler_step(0)
        lr_step0 = t.optimizer.param_groups[0]["lr"]
        self.assertAlmostEqual(lr_step0, 0.0001, places=6)  # 0.01 * 1/100

        t.scheduler_step(200)
        lr_step200 = t.optimizer.param_groups[0]["lr"]
        self.assertLess(lr_step200, 0.01)
        self.assertGreater(lr_step200, 0.0)

    def test_scheduler_step_warmup_cos_decay(self) -> None:
        """scheduler_step applies cosine decay correctly."""
        t = self._make_trainer()
        t._initial_lr = 0.01
        t._scheduler_cfg = type("Cfg", (), {
            "type": "warmup_cos_decay", "warmup_steps": 100,
            "total_steps": 500, "decay_rate": 0.0, "learning_rate_min": 0.001,
        })()

        t.scheduler_step(300)
        lr = t.optimizer.param_groups[0]["lr"]
        self.assertGreater(lr, 0.001)
        self.assertLess(lr, 0.01)

        t.scheduler_step(500)
        lr_end = t.optimizer.param_groups[0]["lr"]
        self.assertAlmostEqual(lr_end, 0.001, places=4)

    def test_fit_epochs_completes(self) -> None:
        """fit_epochs runs through all epochs without error."""
        t = self._make_trainer()
        t.epochs = 3
        t.fit_epochs()
        self.assertEqual(t._epochs_completed, 3)

    def test_update_best_state_better_min(self) -> None:
        """In 'min' mode, lower val_loss is considered better."""
        t = self._make_trainer(monitor="val_loss", monitor_mode="min")
        t.best_metric = 0.5
        result = t.update_best_state({"val_loss": 0.3})
        self.assertEqual(t.best_metric, 0.3)

    def test_update_best_state_worse_min(self) -> None:
        """In 'min' mode, higher val_loss is NOT considered better."""
        t = self._make_trainer(monitor="val_loss", monitor_mode="min")
        t.best_metric = 0.5
        result = t.update_best_state({"val_loss": 0.7})
        self.assertEqual(t.best_metric, 0.5)

    def test_save_and_load_checkpoint(self) -> None:
        """Checkpoint is saved and can be restored."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "test.pt"
            t = self._make_trainer()
            t.best_metric = 0.8
            t.wait = 3
            t.current_epoch = 5
            t.global_step = 1000
            t.save_checkpoint(ckpt_path)
            self.assertTrue(ckpt_path.exists())

            # Load into a new trainer
            t2 = self._make_trainer()
            t2.load_checkpoint(ckpt_path)
            self.assertEqual(t2.best_metric, 0.8)
            self.assertEqual(t2.wait, 3)
            self.assertEqual(t2.current_epoch, 5)
            self.assertEqual(t2.global_step, 1000)

    def test_finalize_epoch_writes_profile(self) -> None:
        """finalize_epoch writes elapsed time and steps/s to profile.txt."""
        with tempfile.TemporaryDirectory() as tmpdir:
            t = self._make_trainer()
            t.set_profile_path(Path(tmpdir))
            t._steps_per_epoch = 100
            t._epoch_start_time = 0.0  # mocked
            t.finalize_epoch(0, {"loss": 0.5}, "Test message")
            profile_file = Path(tmpdir) / "profile.txt"
            self.assertTrue(profile_file.exists())
            content = profile_file.read_text()
            self.assertIn("Test message", content)


if __name__ == "__main__":
    unittest.main()
