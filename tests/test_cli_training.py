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
from gerbil_train.data.shared_bottom_two_tower_dataset import ExplicitDataset
from gerbil_train.data.shared_bottom_two_tower_dataset import ImplicitDataset
from gerbil_train.data.shared_bottom_two_tower_dataset import RankingTestDataset
from gerbil_train.data.shared_bottom_two_tower_dataset import RankingValidationDataset
from gerbil_train.data.shared_bottom_two_tower_dataset import load_ml1m_ratings
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

    def __init__(
        self,
        data_path,
        query_input_dim,
        item_input_dim,
        num_negatives,
        split="full",
    ):
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
        self.split = split

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

    def __init__(self, data_path, query_input_dim, item_input_dim, split="full"):
        """Store shape parameters for generating synthetic explicit samples.

        :param data_path: Unused data path placeholder
        :param query_input_dim: Query feature dimensionality
        :param item_input_dim: Item feature dimensionality
        """
        self.data_path = data_path
        self.query_input_dim = query_input_dim
        self.item_input_dim = item_input_dim
        self.split = split

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


class TinyRankingValidationDataset(Dataset):
    """Tiny ranking-validation dataset for shared-bottom CLI smoke tests."""

    def __init__(self, data_path, query_input_dim, item_input_dim, num_negatives):
        """Store shape parameters for synthetic validation ranking groups.

        :param data_path: Unused data path placeholder
        :param query_input_dim: Query feature dimensionality
        :param item_input_dim: Item feature dimensionality
        :param num_negatives: Number of negatives per validation group
        """
        self.data_path = data_path
        self.query_input_dim = query_input_dim
        self.item_input_dim = item_input_dim
        self.num_negatives = num_negatives

    def __len__(self) -> int:
        """Return the fixed synthetic validation dataset size."""
        return 2

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """Return one synthetic ranking-validation group.

        :param index: Sample index
        :return: Ranking-validation tensors and labels
        """
        num_candidates = self.num_negatives + 1
        query_row = torch.full((self.query_input_dim,), float(index + 1))
        query_features = query_row.repeat(num_candidates, 1)
        item_features = torch.stack(
            [
                torch.full((self.item_input_dim,), float(index + candidate + 2))
                for candidate in range(num_candidates)
            ],
            dim=0,
        )
        labels = torch.zeros(num_candidates, dtype=torch.float32)
        labels[0] = 1.0
        return {
            "query_features": query_features,
            "item_features": item_features,
            "labels": labels,
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

    def test_load_ml1m_ratings_parses_dat_file(self) -> None:
        """Verify that MovieLens ratings.dat can be parsed into a dataframe."""
        with TemporaryDirectory() as tmpdir:
            ratings_path = Path(tmpdir) / "ratings.dat"
            ratings_path.write_text(
                "1::10::4::978300760\n2::20::5::978302109\n",
                encoding="latin-1",
            )

            ratings = load_ml1m_ratings(ratings_path)

            self.assertEqual(
                list(ratings.columns), ["user_id", "item_id", "rating", "timestamp"]
            )
            self.assertEqual(len(ratings), 2)
            self.assertEqual(int(ratings.iloc[0]["user_id"]), 1)
            self.assertEqual(int(ratings.iloc[1]["item_id"]), 20)
            self.assertAlmostEqual(float(ratings.iloc[0]["rating"]), 4.0)

    def test_shared_bottom_datasets_can_read_ml1m_dat_files(self) -> None:
        """Verify that shared-bottom datasets can build samples from raw ml-1m files."""
        with TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            ratings_path = tmp_path / "ratings.dat"
            users_path = tmp_path / "users.dat"
            movies_path = tmp_path / "movies.dat"

            ratings_path.write_text(
                "1::10::4::978300760\n1::20::5::978302109\n2::30::3::978301968\n",
                encoding="latin-1",
            )
            users_path.write_text(
                "1::F::1::10::48067\n2::M::18::20::70072\n",
                encoding="latin-1",
            )
            movies_path.write_text(
                "10::Toy Story (1995)::Animation|Children's|Comedy\n"
                "20::Jumanji (1995)::Adventure|Children's|Fantasy\n"
                "30::Heat (1995)::Action|Crime|Thriller\n",
                encoding="latin-1",
            )

            implicit_dataset = ImplicitDataset(
                data_path=ratings_path,
                query_input_dim=8,
                item_input_dim=8,
                num_negatives=2,
            )
            explicit_dataset = ExplicitDataset(
                data_path=ratings_path,
                query_input_dim=8,
                item_input_dim=8,
            )

            implicit_sample = implicit_dataset[0]
            explicit_sample = explicit_dataset[0]

            self.assertEqual(len(implicit_dataset), 3)
            self.assertEqual(len(explicit_dataset), 3)
            self.assertEqual(tuple(implicit_sample["query_features"].shape), (8,))
            self.assertEqual(tuple(implicit_sample["pos_item_features"].shape), (8,))
            self.assertEqual(tuple(implicit_sample["neg_item_features"].shape), (2, 8))
            self.assertEqual(tuple(explicit_sample["query_features"].shape), (8,))
            self.assertEqual(tuple(explicit_sample["item_features"].shape), (8,))
            self.assertAlmostEqual(float(explicit_sample["label"].item()), 4.0)

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
                    "load_experiment_config",
                    return_value={
                        "data": {
                            "paths": {"dataset": str(dataset_path)},
                        },
                        "model": {
                            "model": {
                                "input_dim": 136,
                                "hidden_dims": [32, 16],
                                "activation": "relu",
                                "dropout": 0.1,
                            }
                        },
                        "train": {
                            "seed": 123,
                            "device": "cpu",
                            "epochs": 1,
                            "loss_name": "mse",
                            "optimizer": {
                                "lr": 1e-3,
                                "weight_decay": 0.0,
                            },
                            "scheduler": {
                                "enabled": True,
                                "factor": 0.5,
                                "patience": 3,
                                "mode": "max",
                            },
                            "gradient": {"clip_grad_norm": None},
                            "checkpoint": {
                                "dir": str(checkpoint_dir),
                                "monitor": "val_ndcg",
                                "mode": "max",
                                "best_checkpoint_path": str(
                                    checkpoint_dir / "best_model_mse.pth"
                                ),
                            },
                            "early_stop": {
                                "enabled": True,
                                "patience": 5,
                            },
                            "logging": {
                                "log_dir": str(checkpoint_dir / "log_dir" / "mse"),
                                "plot_path": str(
                                    checkpoint_dir / "training_curves_mse.png"
                                ),
                            },
                            "evaluation": {"val_k": 5},
                        },
                        "eval": {},
                        "export": {},
                    },
                ),
                patch.object(
                    learning_to_rank_train,
                    "parse_args",
                    return_value=argparse.Namespace(
                        config=Path("unused_experiment.yaml"),
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
                        "validation": {
                            "enabled": True,
                            "num_negatives": 2,
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
                            "explicit_tower": {
                                "hidden_dims": [6],
                                "dropout": 0.0,
                                "activation": "relu",
                                "batch_norm": False,
                            },
                            "implicit_tower": {
                                "hidden_dims": [6],
                                "dropout": 0.0,
                                "activation": "relu",
                                "batch_norm": False,
                            },
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
                        "seed": 42,
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
                        "gradient": {"clip_grad_norm": None},
                        "checkpoint": {
                            "dir": str(tmp_path / "checkpoints"),
                            "save_best_only": True,
                            "save_last": True,
                            "save_every_epoch": False,
                            "monitor": "val_ndcg@2",
                            "mode": "max",
                        },
                        "early_stop": {"enabled": False, "patience": 0},
                        "evaluation": {"validation_k": 2},
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
                    "ImplicitDataset",
                    TinyImplicitDataset,
                ),
                patch.object(
                    shared_bottom_two_tower_train,
                    "ExplicitDataset",
                    TinyExplicitDataset,
                ),
                patch.object(
                    shared_bottom_two_tower_train,
                    "RankingValidationDataset",
                    TinyRankingValidationDataset,
                ),
                patch.object(
                    shared_bottom_two_tower_train,
                    "RankingTestDataset",
                    TinyRankingValidationDataset,
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
            self.assertIn("Final test metrics:", output)
            self.assertTrue((tmp_path / "checkpoints" / "best_model.pth").exists())
            self.assertTrue((tmp_path / "checkpoints" / "last_model.pth").exists())
            self.assertTrue((tmp_path / "checkpoints" / "implicit_loss.txt").exists())
            self.assertTrue((tmp_path / "checkpoints" / "explicit_loss.txt").exists())
            self.assertTrue((tmp_path / "checkpoints" / "training_loss.png").exists())
            self.assertTrue(
                (tmp_path / "checkpoints" / "validation_ndcg@2.txt").exists()
            )
            self.assertTrue(
                (tmp_path / "checkpoints" / "validation_ndcg@2.png").exists()
            )


if __name__ == "__main__":
    unittest.main()
