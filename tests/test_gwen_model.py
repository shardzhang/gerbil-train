from __future__ import annotations

import unittest

import torch

from gerbil_train.config.model_config import FieldEntry, BaseModelConfig
from gerbil_train.models.gwen import GwENMulticlassModel as GwEN


class GwENModelTests(unittest.TestCase):
    """Unit tests for the GwEN model."""

    def _make_config(self) -> BaseModelConfig:
        fields = {
            "user_age": FieldEntry(field_name="user_age", field_index=2, field_type=1, dim=8, emb_size=4),
            "user_gender": FieldEntry(field_name="user_gender", field_index=3, field_type=1, dim=3, emb_size=2),
            "movie_genres": FieldEntry(field_name="movie_genres", field_index=103, field_type=1, dim=19, emb_size=4),
        }
        return BaseModelConfig(
            target_size=10,
            embedding_fields=fields,
            mlp={"hidden_dims": [8, 4], "activation": "relu", "dropout": 0.0, "batch_norm": False},
            field_attention={"enabled": False},
        )

    def _make_batch(self, batch_size: int = 2) -> dict:
        if batch_size == 1:
            return {
                "feature_bags": {
                    "user_age": {
                        "indices": torch.tensor([0, 1], dtype=torch.long),
                        "offsets": torch.tensor([0], dtype=torch.long),
                        "weights": torch.tensor([1.0, 1.0], dtype=torch.float32),
                    },
                    "user_gender": {
                        "indices": torch.tensor([1], dtype=torch.long),
                        "offsets": torch.tensor([0], dtype=torch.long),
                        "weights": torch.tensor([1.0], dtype=torch.float32),
                    },
                    "movie_genres": {
                        "indices": torch.tensor([5, 12], dtype=torch.long),
                        "offsets": torch.tensor([0], dtype=torch.long),
                        "weights": torch.tensor([1.0, 1.0], dtype=torch.float32),
                    },
                }
            }
        return {
            "feature_bags": {
                "user_age": {
                    "indices": torch.tensor([0, 1, 0], dtype=torch.long),
                    "offsets": torch.tensor([0, 2], dtype=torch.long),
                    "weights": torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32),
                },
                "user_gender": {
                    "indices": torch.tensor([1, 0], dtype=torch.long),
                    "offsets": torch.tensor([0, 1], dtype=torch.long),
                    "weights": torch.tensor([1.0, 1.0], dtype=torch.float32),
                },
                "movie_genres": {
                    "indices": torch.tensor([5, 12, 3, 9], dtype=torch.long),
                    "offsets": torch.tensor([0, 2], dtype=torch.long),
                    "weights": torch.tensor([1.0, 1.0, 1.0, 1.0], dtype=torch.float32),
                },
            }
        }

    def test_forward_shape(self) -> None:
        """Verify forward pass returns correct logits shape."""
        model = GwEN(self._make_config())
        batch = self._make_batch(batch_size=2)
        logits = model(batch["feature_bags"])
        self.assertEqual(tuple(logits.shape), (2, 10))

    def test_encode_shape(self) -> None:
        """Verify encode returns the MLP hidden state."""
        model = GwEN(self._make_config())
        batch = self._make_batch(batch_size=2)
        hidden = model.encode(batch["feature_bags"])
        # hidden_dim should be the last MLP layer output = 4
        self.assertEqual(tuple(hidden.shape), (2, 4))

    def test_forward_with_attention(self) -> None:
        """Verify forward pass works with attention enabled."""
        config = self._make_config()
        config.field_attention = {"enabled": True}
        model = GwEN(config)
        batch = self._make_batch(batch_size=2)
        logits = model(batch["feature_bags"])
        self.assertEqual(tuple(logits.shape), (2, 10))

    def test_batch_size_one(self) -> None:
        """Verify forward pass works with batch size 1."""
        model = GwEN(self._make_config())
        batch = self._make_batch(batch_size=1)
        logits = model(batch["feature_bags"])
        self.assertEqual(tuple(logits.shape), (1, 10))


if __name__ == "__main__":
    unittest.main()
