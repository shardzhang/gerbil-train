from __future__ import annotations

import unittest

import torch

from gerbil_train.models.deepfm import DeepFM


class DeepFMModelTests(unittest.TestCase):
    """Unit tests for the DeepFM model."""

    def test_deepfm_forward_with_mapping_sparse_features(self) -> None:
        """Verify that DeepFM accepts mapping-based sparse inputs."""
        model = DeepFM(
            {
                "dense_input_dim": 3,
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
        )

        outputs = model(
            dense_features=torch.randn(2, 3),
            sparse_features={
                "user_id": torch.tensor([1, 2]),
                "item_id": torch.tensor([3, 4]),
            },
        )

        self.assertEqual(tuple(outputs.shape), (2,))

    def test_deepfm_forward_with_tensor_sparse_features_and_sigmoid(self) -> None:
        """Verify that DeepFM accepts tensor sparse inputs and applies sigmoid."""
        model = DeepFM(
            {
                "dense_input_dim": 0,
                "embedding_dim": 4,
                "sparse_fields": {
                    "user_id": {"vocab_size": 10},
                    "item_id": {"vocab_size": 20},
                },
                "deep": {
                    "hidden_dims": [8],
                    "activation": "relu",
                    "dropout": 0.0,
                    "batch_norm": False,
                },
                "output": {"activation": "sigmoid"},
            }
        )

        outputs = model(
            sparse_features=torch.tensor(
                [
                    [1, 3],
                    [2, 4],
                ]
            )
        )

        self.assertEqual(tuple(outputs.shape), (2,))
        self.assertTrue(torch.all(outputs >= 0.0).item())
        self.assertTrue(torch.all(outputs <= 1.0).item())


if __name__ == "__main__":
    unittest.main()
