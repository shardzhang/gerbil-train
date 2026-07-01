from __future__ import annotations

import unittest

import torch

from gerbil_train.config.model_config import YouTubeDNNModelConfig, FieldEntry
from gerbil_train.models.youtube_dnn import YouTubeDNN


class YouTubeDNNModelTests(unittest.TestCase):
    """Unit tests for the YouTubeDNN model."""

    def _make_config(self) -> YouTubeDNNModelConfig:
        fields = {
            "user_id": FieldEntry(field_name="user_id", field_index=1, field_type=1, dim=10, emb_size=8),
            "watch_history": FieldEntry(field_name="watch_history", field_index=2, field_type=1, dim=30, emb_size=8),
        }
        return YouTubeDNNModelConfig(
            target_size=10,
            embedding_fields=fields,
            behavior_fields=["watch_history"],
            mlp={"hidden_dims": [8, 4], "activation": "relu", "dropout": 0.0, "batch_norm": False},
        )

    def _make_bags(self, batch_size: int = 2) -> dict:
        return {
            "feature_bags": {
                "user_id": {
                    "indices": torch.tensor([0, 1, 0, 1], dtype=torch.long),
                    "offsets": torch.tensor([0, 2], dtype=torch.long),
                    "weights": torch.tensor([1.0, 1.0, 1.0, 1.0], dtype=torch.float32),
                },
                "watch_history": {
                    "indices": torch.tensor([1, 3, 5, 2, 4, 6], dtype=torch.long),
                    "offsets": torch.tensor([0, 3], dtype=torch.long),
                    "weights": torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=torch.float32),
                },
            }
        }

    def test_forward_shape(self) -> None:
        """Forward returns logits of shape [batch_size, target_size]."""
        model = YouTubeDNN(self._make_config())
        batch = self._make_bags(batch_size=2)
        logits = model(batch["feature_bags"])
        self.assertEqual(tuple(logits.shape), (2, 10))

    def test_encode_shape(self) -> None:
        """Encode returns user embedding of shape [batch_size, final_hidden_dim]."""
        model = YouTubeDNN(self._make_config())
        batch = self._make_bags(batch_size=2)
        user_emb = model.encode(batch["feature_bags"])
        self.assertEqual(tuple(user_emb.shape), (2, 4))

    def test_head_bias_false(self) -> None:
        """Default head.bias is None."""
        model = YouTubeDNN(self._make_config())
        self.assertIsNone(model.head.bias)

    def test_behavior_mode_mean(self) -> None:
        """Behavior field uses mode='mean'."""
        model = YouTubeDNN(self._make_config())
        bag = model.embedding_bags[str(2)]
        self.assertEqual(bag.mode, "mean")

    def test_example_age(self) -> None:
        """Example age is processed with log(age+1)."""
        cfg = self._make_config()
        cfg.example_age_field = "age"
        cfg.embedding_fields["age"] = FieldEntry(
            field_name="age", field_index=5, field_type=1, dim=8, emb_size=1,
        )
        cfg.behavior_fields = [f for f in cfg.behavior_fields if f != "watch_history"]
        model = YouTubeDNN(cfg)
        bags = self._make_bags()["feature_bags"]
        bags["age"] = {
            "indices": torch.tensor([0, 0], dtype=torch.long),
            "offsets": torch.tensor([0, 1], dtype=torch.long),
            "weights": torch.tensor([10.0, 5.0], dtype=torch.float32),
        }
        logits = model(bags)
        self.assertEqual(tuple(logits.shape), (2, 10))


if __name__ == "__main__":
    unittest.main()
