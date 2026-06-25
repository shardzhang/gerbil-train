from __future__ import annotations

import unittest

import torch

from gerbil_train.config.model_config import DeepFMModelConfig, FieldEntry
from gerbil_train.models.deepfm import DeepFM


class DeepFMModelTests(unittest.TestCase):
    """Unit tests for the DeepFM model."""

    def _make_config(self, output_activation: str = "none") -> DeepFMModelConfig:
        fields = {
            "user_id": FieldEntry(field_name="user_id", field_index=1, field_type=1, dim=10, emb_size=4),
            "item_id": FieldEntry(field_name="item_id", field_index=2, field_type=1, dim=20, emb_size=4),
        }
        return DeepFMModelConfig(
            target_size=0,
            embedding_fields=fields,
            mlp={"hidden_dims": [8, 4], "activation": "relu", "dropout": 0.0, "batch_norm": False},
            output={"activation": output_activation},
        )

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
        model = DeepFM(self._make_config(output_activation="sigmoid"))
        outputs = model(self._make_bags()["feature_bags"])
        self.assertEqual(tuple(outputs.shape), (2,))
        self.assertTrue(torch.all(outputs >= 0.0).item())
        self.assertTrue(torch.all(outputs <= 1.0).item())


if __name__ == "__main__":
    unittest.main()
