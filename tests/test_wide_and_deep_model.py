from __future__ import annotations

import unittest

import torch

from gerbil_train.config.model_config import WideAndDeepModelConfig, FieldEntry
from gerbil_train.models.wide_and_deep import WideAndDeep


class WideAndDeepModelTests(unittest.TestCase):
    """Unit tests for the Wide & Deep model."""

    def _make_config(self, extra: dict | None = None) -> WideAndDeepModelConfig:
        fields = {
            "user_id": FieldEntry(field_name="user_id", field_index=1, field_type=1, dim=10, emb_size=4),
            "item_id": FieldEntry(field_name="item_id", field_index=2, field_type=1, dim=20, emb_size=4),
            **({} if extra is None else extra),
        }
        return WideAndDeepModelConfig(
            target_size=0,
            embedding_fields=fields,
            mlp={"hidden_dims": [8], "activation": "relu", "dropout": 0.0, "batch_norm": False},
            output={"activation": "none"},
        )

    def _make_bags(self, extra: dict | None = None) -> dict:
        bags = {
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
            **(extra or {}),
        }
        return {"feature_bags": bags}

    def test_forward(self) -> None:
        """Forward pass returns correct output shape."""
        model = WideAndDeep(self._make_config())
        outputs = model(self._make_bags()["feature_bags"])
        self.assertEqual(tuple(outputs.shape), (2,))

    def test_sigmoid_range(self) -> None:
        """Sigmoid output is in [0, 1]."""
        model = WideAndDeep(self._make_config())
        outputs = model(self._make_bags()["feature_bags"])
        self.assertTrue(torch.all(outputs >= 0.0).item())
        self.assertTrue(torch.all(outputs <= 1.0).item())

    def test_wide_only_field(self) -> None:
        """Field with wide=True, deep=False only affects linear term."""
        cfg = self._make_config({
            "gender": FieldEntry(field_name="gender", field_index=3, field_type=1, dim=5, emb_size=4,
                                 wide=True, deep=False),
        })
        model = WideAndDeep(cfg)
        outputs = model(self._make_bags({
            "gender": {
                "indices": torch.tensor([0, 1], dtype=torch.long),
                "offsets": torch.tensor([0, 1], dtype=torch.long),
                "weights": torch.tensor([1.0, 1.0], dtype=torch.float32),
            },
        })["feature_bags"])
        self.assertEqual(tuple(outputs.shape), (2,))

    def test_deep_only_field(self) -> None:
        """Field with wide=False, deep=True only affects MLP term."""
        cfg = self._make_config({
            "score": FieldEntry(field_name="score", field_index=4, field_type=0, dim=1, emb_size=4,
                                concat_type="direct", wide=False, deep=True),
        })
        model = WideAndDeep(cfg)
        outputs = model(self._make_bags({
            "score": {
                "indices": torch.tensor([0, 0], dtype=torch.long),
                "offsets": torch.tensor([0, 1], dtype=torch.long),
                "weights": torch.tensor([0.5, 0.8], dtype=torch.float32),
            },
        })["feature_bags"])
        self.assertEqual(tuple(outputs.shape), (2,))


if __name__ == "__main__":
    unittest.main()
