from __future__ import annotations

import unittest

import torch

from gerbil_train.config.model_config import DIENModelConfig, FieldEntry
from gerbil_train.models.dien import DIEN


class DIENModelTests(unittest.TestCase):
    """Unit tests for the DIEN model."""

    def _make_config(self) -> DIENModelConfig:
        fields = {
            "user_id": FieldEntry(field_name="user_id", field_index=1, field_type=1, dim=10, emb_size=4),
            "movie_id": FieldEntry(field_name="movie_id", field_index=101, field_type=1, dim=20, emb_size=4),
            "history": FieldEntry(field_name="history", field_index=301, field_type=1, dim=30, emb_size=4),
        }
        return DIENModelConfig(
            target_size=10,
            embedding_fields=fields,
            behavior_fields=["history"],
            target_fields=["movie_id"],
            mlp={"hidden_dims": [8, 4], "activation": "relu", "dropout": 0.0, "batch_norm": False},
            interest_extractor={"hidden_size": 4, "num_layers": 1},
            local_activation_unit={"hidden_dims": [4], "bias": [True], "batch_norm": False, "activation": "relu"},
        )

    def _make_bags(self, batch_size: int = 2) -> dict:
        """Create feature bags with one behavior sequence field."""
        if batch_size == 1:
            return {
                "feature_bags": {
                    "user_id": {
                        "indices": torch.tensor([0], dtype=torch.long),
                        "offsets": torch.tensor([0], dtype=torch.long),
                        "weights": torch.tensor([1.0], dtype=torch.float32),
                    },
                    "movie_id": {
                        "indices": torch.tensor([3], dtype=torch.long),
                        "offsets": torch.tensor([0], dtype=torch.long),
                        "weights": torch.tensor([1.0], dtype=torch.float32),
                    },
                    "history": {
                        "indices": torch.tensor([1, 2, 3], dtype=torch.long),
                        "offsets": torch.tensor([0], dtype=torch.long),
                        "weights": torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32),
                    },
                }
            }
        return {
            "feature_bags": {
                "user_id": {
                    "indices": torch.tensor([0, 1, 0, 1], dtype=torch.long),
                    "offsets": torch.tensor([0, 2], dtype=torch.long),
                    "weights": torch.tensor([1.0, 1.0, 1.0, 1.0], dtype=torch.float32),
                },
                "movie_id": {
                    "indices": torch.tensor([3, 5], dtype=torch.long),
                    "offsets": torch.tensor([0, 1], dtype=torch.long),
                    "weights": torch.tensor([1.0, 1.0], dtype=torch.float32),
                },
                "history": {
                    "indices": torch.tensor([1, 2, 3, 4, 5, 6], dtype=torch.long),
                    "offsets": torch.tensor([0, 3], dtype=torch.long),
                    "weights": torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=torch.float32),
                },
            }
        }

    def test_forward_shape(self) -> None:
        """Forward returns sigmoid of shape [batch_size]."""
        model = DIEN(self._make_config())
        batch = self._make_bags(batch_size=2)
        out = model(batch["feature_bags"])
        self.assertEqual(tuple(out.shape), (2,))
        self.assertTrue(torch.all(out >= 0.0).item())
        self.assertTrue(torch.all(out <= 1.0).item())

    def test_forward_with_aux_shape(self) -> None:
        """forward_with_aux returns (sigmoid, aux_logits_dict)."""
        model = DIEN(self._make_config())
        batch = self._make_bags(batch_size=2)
        sigmoids, aux_logits = model.forward_with_aux(batch["feature_bags"])
        self.assertEqual(tuple(sigmoids.shape), (2,))
        self.assertIsInstance(aux_logits, dict)

    def test_forward_batch_size_one(self) -> None:
        """Works with a single sample."""
        model = DIEN(self._make_config())
        batch = self._make_bags(batch_size=1)
        out = model(batch["feature_bags"])
        self.assertEqual(tuple(out.shape), (1,))

    def test_no_behavior_fields(self) -> None:
        """DIEN with empty behavior_fields raises ValueError."""
        cfg = self._make_config()
        cfg.behavior_fields = []
        with self.assertRaises(ValueError):
            DIEN(cfg)


if __name__ == "__main__":
    unittest.main()
