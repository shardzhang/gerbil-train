from __future__ import annotations

import unittest

import torch

from gerbil_train.config.model_config import BaseModelConfig, FieldEntry
from gerbil_train.models.ftrl import FTRLModel


class FTRLModelTests(unittest.TestCase):
    """Unit tests for the FTRL model."""

    def _make_config(self) -> BaseModelConfig:
        fields = {
            "user_id": FieldEntry(field_name="user_id", field_index=1, field_type=1, dim=10, emb_size=4),
            "item_id": FieldEntry(field_name="item_id", field_index=2, field_type=1, dim=20, emb_size=4),
        }
        return BaseModelConfig(
            target_size=0,
            embedding_fields=fields,
            mlp={"hidden_dims": [8], "activation": "relu", "dropout": 0.0, "batch_norm": False},
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
        """Forward returns sigmoid of shape [batch_size]."""
        model = FTRLModel(self._make_config())
        out = model(self._make_bags()["feature_bags"])
        self.assertEqual(tuple(out.shape), (2,))

    def test_sigmoid_range(self) -> None:
        """Sigmoid output is in [0, 1]."""
        model = FTRLModel(self._make_config())
        out = model(self._make_bags()["feature_bags"])
        self.assertTrue(torch.all(out >= 0.0).item())
        self.assertTrue(torch.all(out <= 1.0).item())

    def test_ftrl_optimizer_step(self) -> None:
        """FTRL optimizer can take a step without error."""
        from gerbil_train.optimizers.ftrl import FTRL
        model = FTRLModel(self._make_config())
        opt = FTRL(model.parameters(), alpha=0.1, beta=1.0, lambda1=0.5, lambda2=0.5)
        out = model(self._make_bags()["feature_bags"])
        loss = (out - torch.ones_like(out)).pow(2).mean()
        loss.backward()
        opt.step()


if __name__ == "__main__":
    unittest.main()
