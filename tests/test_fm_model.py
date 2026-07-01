from __future__ import annotations

import unittest

import torch

from gerbil_train.config.model_config import BaseModelConfig, FieldEntry
from gerbil_train.models.fm import FM


class FMModelTests(unittest.TestCase):
    """Unit tests for the FM model."""

    def _make_config(self) -> BaseModelConfig:
        fields = {
            "user_id": FieldEntry(field_name="user_id", field_index=1, field_type=1, dim=10, emb_size=4),
            "item_id": FieldEntry(field_name="item_id", field_index=2, field_type=1, dim=20, emb_size=4),
        }
        return BaseModelConfig(
            target_size=0,
            embedding_fields=fields,
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

    def test_forward_shape(self) -> None:
        model = FM(self._make_config())
        out = model(self._make_bags()["feature_bags"])
        self.assertEqual(tuple(out.shape), (2,))

    def test_sigmoid_range(self) -> None:
        model = FM(self._make_config())
        out = model(self._make_bags()["feature_bags"])
        self.assertTrue(torch.all(out >= 0.0).item())
        self.assertTrue(torch.all(out <= 1.0).item())

    def test_different_emb_size_raises(self) -> None:
        with self.assertRaises(ValueError):
            fields = {
                "a": FieldEntry(field_name="a", field_index=1, field_type=1, dim=10, emb_size=4),
                "b": FieldEntry(field_name="b", field_index=2, field_type=1, dim=10, emb_size=8),
            }
            FM(BaseModelConfig(target_size=0, embedding_fields=fields))


if __name__ == "__main__":
    unittest.main()
