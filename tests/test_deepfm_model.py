from __future__ import annotations

import unittest

import torch

from gerbil_train.models.deepfm import DeepFM


class DeepFMModelTests(unittest.TestCase):
    """Unit tests for the DeepFM model."""

    def _make_config(self) -> dict:
        return {
            "field_names": ["user_id", "item_id"],
            "embedding_dim": 4,
            "sparse_fields": {
                "user_id": {"vocab_size": 10},
                "item_id": {"vocab_size": 20},
            },
            "deep": {
                "hidden_dims": [8, 4],
                "activation": "relu",
                "dropout": 0.0,
                "batch_norm": False,
            },
            "output": {"activation": None},
        }

    def _make_bags(self) -> dict:
        return {
            "feature_bags": {
                "user_id": {
                    "indices": torch.tensor([1, 2], dtype=torch.long),
                    "offsets": torch.tensor([0, 1], dtype=torch.long),
                    "weights": torch.tensor([1.0, 1.0], dtype=torch.float32),
                },
                "item_id": {
                    "indices": torch.tensor([3, 4], dtype=torch.long),
                    "offsets": torch.tensor([0, 1], dtype=torch.long),
                    "weights": torch.tensor([1.0, 1.0], dtype=torch.float32),
                },
            }
        }

    def test_deepfm_forward(self) -> None:
        """Verify forward pass returns correct output shape."""
        model = DeepFM(self._make_config())
        outputs = model(self._make_bags()["feature_bags"])
        self.assertEqual(tuple(outputs.shape), (2,))

    def test_deepfm_forward_with_sigmoid(self) -> None:
        """Verify sigmoid output is in [0, 1] range."""
        config = self._make_config()
        config["output"] = {"activation": "sigmoid"}
        model = DeepFM(config)
        outputs = model(self._make_bags()["feature_bags"])
        self.assertEqual(tuple(outputs.shape), (2,))
        self.assertTrue(torch.all(outputs >= 0.0).item())
        self.assertTrue(torch.all(outputs <= 1.0).item())


if __name__ == "__main__":
    unittest.main()
