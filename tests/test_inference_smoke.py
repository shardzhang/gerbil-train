"""Inference smoke tests: Predictor load, predict, evaluate."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn as nn

from gerbil_train.inference.predictor import Predictor
from gerbil_train.inference.result_writer import write_results


class DummyModel(nn.Module):
    """Model that returns random sigmoid values."""
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 1)

    def forward(self, feature_bags: dict) -> torch.Tensor:
        batch = feature_bags.get("_batch_size", 2)
        return torch.sigmoid(torch.randn(batch)).squeeze(-1)

    def encode(self, feature_bags: dict) -> torch.Tensor:
        batch = feature_bags.get("_batch_size", 2)
        return torch.randn(batch, 8)


class PredictorSmokeTests(unittest.TestCase):
    """Smoke tests for the Predictor class."""

    def test_predict_returns_results(self) -> None:
        """predict() returns a list of dicts with score/label."""
        model = DummyModel()
        predictor = Predictor(model, device="cpu")
        results = predictor.predict(self._make_dataloader())
        self.assertGreater(len(results), 0)
        self.assertIn("user_id", results[0])
        self.assertIn("score", results[0])
        self.assertIn("label", results[0])

    def test_evaluate_returns_metrics(self) -> None:
        """evaluate() returns a dict of metric names to values."""
        model = DummyModel()
        predictor = Predictor(model, device="cpu")
        metrics = predictor.evaluate(self._make_dataloader())
        self.assertIn("auc", metrics)
        self.assertGreaterEqual(metrics["auc"], 0.0)
        self.assertLessEqual(metrics["auc"], 1.0)

    def test_predict_and_eval_writes_file(self) -> None:
        """predict_and_eval() with output path writes predictions to disk."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "pred.tsv"
            model = DummyModel()
            predictor = Predictor(model, device="cpu")
            metrics = predictor.predict_and_eval(
                self._make_dataloader(), output_path=output,
            )
            self.assertTrue(output.exists())
            self.assertIn("auc", metrics)

    def test_load_checkpoint(self) -> None:
        """load_checkpoint restores model weights."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt = Path(tmpdir) / "model.pth"
            model = DummyModel()
            torch.save({"model_state_dict": model.state_dict()}, ckpt)
            predictor = Predictor(DummyModel(), device="cpu")
            predictor.load_checkpoint(ckpt)
            results = predictor.predict(self._make_dataloader())
            self.assertGreater(len(results), 0)

    def test_load_checkpoint_orig_mod_prefix(self) -> None:
        """load_checkpoint strips _orig_mod. prefix from torch.compile'd checkpoints."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt = Path(tmpdir) / "compiled.pth"
            model = DummyModel()
            state = {"model_state_dict": {f"_orig_mod.{k}": v for k, v in model.state_dict().items()}}
            torch.save(state, ckpt)
            predictor = Predictor(DummyModel(), device="cpu")
            predictor.load_checkpoint(ckpt)
            results = predictor.predict(self._make_dataloader())
            self.assertGreater(len(results), 0)

    def test_write_results_tsv(self) -> None:
        """write_results produces a valid TSV file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "out.tsv"
            results = [
                {"user_id": 1, "score": 0.95, "label": 1},
                {"user_id": 2, "score": 0.10, "label": 0},
            ]
            write_results(results, output, fmt="tsv")
            content = output.read_text().strip().split("\n")
            self.assertEqual(len(content), 3)  # header + 2 lines
            self.assertIn("user_id", content[0])

    def test_write_results_json(self) -> None:
        """write_results produces a valid JSON file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "out.json"
            results = [{"user_id": 1, "score": 0.95, "label": 1}]
            write_results(results, output, fmt="json")
            import json
            data = json.loads(output.read_text())
            self.assertEqual(len(data), 1)
            self.assertEqual(data[0]["user_id"], 1)

    @staticmethod
    def _make_dataloader() -> list:
        """Return a mock list of batches (simulates DataLoader iteration)."""
        return [
            {
                "feature_bags": {"_batch_size": 2},
                "targets": torch.tensor([1, 0]),
            },
            {
                "feature_bags": {"_batch_size": 2},
                "targets": torch.tensor([0, 1]),
            },
        ]


if __name__ == "__main__":
    unittest.main()
