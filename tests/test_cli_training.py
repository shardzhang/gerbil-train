from __future__ import annotations

import argparse
import io
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import torch
import yaml
from torch.utils.data import Dataset

from gerbil_train.cli import learning_to_rank_train
from gerbil_train.cli import shared_bottom_two_tower_train
from gerbil_train.data.learning_to_rank_dataset import build_ltr_dataloaders
from gerbil_train.data.learning_to_rank_dataset import LearningToRankDataset
from gerbil_train.data.learning_to_rank_dataset import load_mslrweb10k_groups
from gerbil_train.trainer import learning_to_rank_trainer


class DummyTqdm:
    """Minimal ``tqdm`` stand-in used by tests."""

    def __init__(self, iterable, **kwargs):
        """Store the wrapped iterable and ignore progress-bar options.

        :param iterable: Iterable wrapped by the dummy progress bar
        :param kwargs: Ignored keyword arguments from ``tqdm`` callers
        """
        self.iterable = iterable
        self.options = kwargs

    def __iter__(self):
        """Iterate over the wrapped iterable."""
        return iter(self.iterable)

    def set_postfix(self, **kwargs):
        """Ignore postfix updates emitted by the code under test.

        :param kwargs: Ignored postfix key-value pairs
        """
        self.postfix = kwargs


class TinyImplicitDataset(Dataset):
    """Tiny implicit-feedback dataset for CLI smoke tests."""

    def __init__(self, data_path, query_input_dim, item_input_dim, num_negatives):
        """Store shape parameters for generating synthetic implicit samples.

        :param data_path: Unused data path placeholder
        :param query_input_dim: Query feature dimensionality
        :param item_input_dim: Item feature dimensionality
        :param num_negatives: Number of negatives per sample
        """
        self.data_path = data_path
        self.query_input_dim = query_input_dim
        self.item_input_dim = item_input_dim
        self.num_negatives = num_negatives

    def __len__(self) -> int:
        """Return the fixed synthetic dataset size."""
        return 2

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """Return one synthetic implicit-feedback training example.

        :param index: Sample index
        :return: Dictionary containing implicit-feedback tensors
        """
        query_features = torch.full((self.query_input_dim,), float(index + 1))
        pos_item_features = torch.full((self.item_input_dim,), float(index + 2))
        neg_item_features = torch.full(
            (self.num_negatives, self.item_input_dim),
            float(index + 3),
        )
        return {
            "query_features": query_features,
            "pos_item_features": pos_item_features,
            "neg_item_features": neg_item_features,
        }


class TinyExplicitDataset(Dataset):
    """Tiny explicit-feedback dataset for CLI smoke tests."""

    def __init__(self, data_path, query_input_dim, item_input_dim):
        """Store shape parameters for generating synthetic explicit samples.

        :param data_path: Unused data path placeholder
        :param query_input_dim: Query feature dimensionality
        :param item_input_dim: Item feature dimensionality
        """
        self.data_path = data_path
        self.query_input_dim = query_input_dim
        self.item_input_dim = item_input_dim

    def __len__(self) -> int:
        """Return the fixed synthetic dataset size."""
        return 2

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """Return one synthetic explicit-feedback training example.

        :param index: Sample index
        :return: Dictionary containing explicit-feedback tensors
        """
        query_features = torch.full((self.query_input_dim,), float(index + 1))
        item_features = torch.full((self.item_input_dim,), float(index + 2))
        label = torch.tensor(float(index) / 2.0)
        return {
            "query_features": query_features,
            "item_features": item_features,
            "label": label,
        }


class CliTrainingTests(unittest.TestCase):
    """Smoke tests for the training CLI entrypoints."""

    def test_learning_to_rank_groups_are_dataset_instances(self) -> None:
        """Verify that MSLR split loading returns dataset objects."""
        with TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            dataset_path = tmp_path / "tiny_ltr.pt"

            dataset = {
                "train": {
                    1: (
                        np.array(
                            [
                                np.linspace(0.1, 1.0, 136),
                                np.linspace(0.2, 1.1, 136),
                            ],
                            dtype=np.float32,
                        ),
                        np.array([1, 0], dtype=np.int64),
                    )
                },
                "vali": {
                    2: (
                        np.array(
                            [
                                np.linspace(0.3, 1.2, 136),
                                np.linspace(0.4, 1.3, 136),
                            ],
                            dtype=np.float32,
                        ),
                        np.array([0, 1], dtype=np.int64),
                    )
                },
                "test": {
                    3: (
                        np.array(
                            [
                                np.linspace(0.5, 1.4, 136),
                                np.linspace(0.6, 1.5, 136),
                            ],
                            dtype=np.float32,
                        ),
                        np.array([1, 0], dtype=np.int64),
                    )
                },
            }
            torch.save(dataset, dataset_path)

            train_groups, val_groups, test_groups = load_mslrweb10k_groups(dataset_path)

            self.assertIsInstance(train_groups, LearningToRankDataset)
            self.assertIsInstance(val_groups, LearningToRankDataset)
            self.assertIsInstance(test_groups, LearningToRankDataset)
            self.assertEqual(len(train_groups), 1)
            self.assertIn("X", train_groups[0])
            self.assertIn("y", train_groups[0])

    def test_learning_to_rank_dataloaders_are_buildable(self) -> None:
        """Verify that LTR dataloaders can be constructed from a tiny dataset."""
        with TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            dataset_path = tmp_path / "tiny_ltr.pt"

            dataset = {
                "train": {
                    1: (
                        np.array(
                            [
                                np.linspace(0.1, 1.0, 136),
                                np.linspace(0.2, 1.1, 136),
                            ],
                            dtype=np.float32,
                        ),
                        np.array([1, 0], dtype=np.int64),
                    )
                },
                "vali": {
                    2: (
                        np.array(
                            [
                                np.linspace(0.3, 1.2, 136),
                                np.linspace(0.4, 1.3, 136),
                            ],
                            dtype=np.float32,
                        ),
                        np.array([0, 1], dtype=np.int64),
                    )
                },
                "test": {
                    3: (
                        np.array(
                            [
                                np.linspace(0.5, 1.4, 136),
                                np.linspace(0.6, 1.5, 136),
                            ],
                            dtype=np.float32,
                        ),
                        np.array([1, 0], dtype=np.int64),
                    )
                },
            }
            torch.save(dataset, dataset_path)

            train_loader, val_loader, test_loader = build_ltr_dataloaders(dataset_path)

            self.assertEqual(len(train_loader), 1)
            self.assertEqual(len(val_loader), 1)
            self.assertEqual(len(test_loader), 1)
            batch = next(iter(train_loader))
            self.assertIn("X", batch)
            self.assertIn("y", batch)

    def test_learning_to_rank_cli_runs_with_tiny_dataset(self) -> None:
        """Verify that the LTR CLI can run end-to-end on a tiny dataset."""
        with TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            dataset_path = tmp_path / "tiny_ltr.pt"
            checkpoint_dir = tmp_path / "checkpoints"

            dataset = {
                "train": {
                    1: (
                        np.array(
                            [
                                np.linspace(0.1, 1.0, 136),
                                np.linspace(0.2, 1.1, 136),
                                np.linspace(0.3, 1.2, 136),
                            ],
                            dtype=np.float32,
                        ),
                        np.array([2, 1, 0], dtype=np.int64),
                    )
                },
                "vali": {
                    2: (
                        np.array(
                            [
                                np.linspace(0.4, 1.3, 136),
                                np.linspace(0.5, 1.4, 136),
                                np.linspace(0.6, 1.5, 136),
                            ],
                            dtype=np.float32,
                        ),
                        np.array([2, 0, 1], dtype=np.int64),
                    )
                },
                "test": {
                    3: (
                        np.array(
                            [
                                np.linspace(0.7, 1.6, 136),
                                np.linspace(0.8, 1.7, 136),
                                np.linspace(0.9, 1.8, 136),
                            ],
                            dtype=np.float32,
                        ),
                        np.array([1, 2, 0], dtype=np.int64),
                    )
                },
            }
            torch.save(dataset, dataset_path)

            with (
                patch.object(torch.cuda, "is_available", return_value=False),
                patch.object(learning_to_rank_trainer, "tqdm", DummyTqdm),
                patch.object(
                    learning_to_rank_train,
                    "parse_args",
                    return_value=argparse.Namespace(
                        loss="mse",
                        dataset_path=dataset_path,
                        epochs=1,
                        seed=123,
                        checkpoint_dir=checkpoint_dir,
                        monitor="val_ndcg",
                        monitor_mode="max",
                        patience=5,
                    ),
                ),
            ):
                learning_to_rank_train.main()

            self.assertTrue((checkpoint_dir / "best_model_mse.pth").exists())
            self.assertTrue((checkpoint_dir / "training_curves_mse.png").exists())
            self.assertTrue((checkpoint_dir / "training_curves_mse_loss.txt").exists())
            self.assertTrue(
                (checkpoint_dir / "training_curves_mse_metric.txt").exists()
            )

    def test_shared_bottom_two_tower_cli_runs_with_tiny_datasets(self) -> None:
        """Verify that the shared-bottom two-tower CLI runs on tiny datasets."""
        with TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            data_cfg_path = tmp_path / "data.yaml"
            model_cfg_path = tmp_path / "model.yaml"
            train_cfg_path = tmp_path / "train.yaml"
            eval_cfg_path = tmp_path / "eval.yaml"
            export_cfg_path = tmp_path / "export.yaml"
            experiment_cfg_path = tmp_path / "experiment.yaml"

            data_cfg_path.write_text(
                yaml.safe_dump(
                    {
                        "loader": {
                            "train_batch_size": 2,
                            "num_workers": 0,
                            "pin_memory": False,
                            "drop_last": False,
                        },
                        "sampling": {
                            "implicit": {
                                "negative_sampling": True,
                                "num_negatives": 2,
                            }
                        },
                        "paths": {
                            "train_implicit": "unused_implicit.pt",
                            "train_explicit": "unused_explicit.pt",
                        },
                    }
                ),
                encoding="utf-8",
            )
            model_cfg_path.write_text(
                yaml.safe_dump(
                    {
                        "model": {
                            "query_input_dim": 4,
                            "item_input_dim": 5,
                            "shared_bottom": {
                                "hidden_dims": [8],
                                "dropout": 0.0,
                                "activation": "relu",
                                "batch_norm": False,
                            },
                            "explicit_tower": {"hidden_dims": [6]},
                            "implicit_tower": {"hidden_dims": [6]},
                            "embedding_dim": 3,
                            "normalize_embedding": False,
                        }
                    }
                ),
                encoding="utf-8",
            )
            train_cfg_path.write_text(
                yaml.safe_dump(
                    {
                        "device": "cpu",
                        "optimizer": {
                            "implicit": {"lr": 1e-3, "weight_decay": 0.0},
                            "explicit": {"lr": 1e-3},
                        },
                        "trainer": {
                            "implicit_epochs": 1,
                            "explicit_epochs": 1,
                            "gradient_clip_norm": None,
                        },
                    }
                ),
                encoding="utf-8",
            )
            eval_cfg_path.write_text(yaml.safe_dump({}), encoding="utf-8")
            export_cfg_path.write_text(yaml.safe_dump({}), encoding="utf-8")
            experiment_cfg_path.write_text(
                yaml.safe_dump(
                    {
                        "data": str(data_cfg_path),
                        "model": str(model_cfg_path),
                        "train": str(train_cfg_path),
                        "eval": str(eval_cfg_path),
                        "export": str(export_cfg_path),
                    }
                ),
                encoding="utf-8",
            )

            stdout = io.StringIO()
            with (
                patch.object(torch.cuda, "is_available", return_value=False),
                patch.object(
                    shared_bottom_two_tower_train,
                    "SharedBottomTwoTowerImplicitDataset",
                    TinyImplicitDataset,
                ),
                patch.object(
                    shared_bottom_two_tower_train,
                    "SharedBottomTwoTowerExplicitDataset",
                    TinyExplicitDataset,
                ),
                patch.object(
                    shared_bottom_two_tower_train,
                    "parse_args",
                    return_value=argparse.Namespace(config=str(experiment_cfg_path)),
                ),
                redirect_stdout(stdout),
            ):
                shared_bottom_two_tower_train.main()

            output = stdout.getvalue()
            self.assertIn("[implicit] epoch=1", output)
            self.assertIn("[explicit] epoch=1", output)


if __name__ == "__main__":
    unittest.main()
