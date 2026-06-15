from __future__ import annotations

import argparse
import io
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import torch
import yaml
from torch.utils.data import Dataset

from gerbil_train.cli import deepfm_train


class TinyDeepFMDataset(Dataset):
    """Tiny pointwise dataset for DeepFM CLI smoke tests."""

    def __init__(self, data_path, split="train"):
        self.data_path = data_path
        self.split = split

    def __len__(self) -> int:
        return 2

    def __getitem__(self, index: int):
        return {
            "dense_features": torch.zeros(0, dtype=torch.float32),
            "sparse_features": torch.tensor(
                [index + 1, index + 2, 1, 18, 5, 24, 3],
                dtype=torch.long,
            ),
            "label": torch.tensor(1.0 if index == 0 else 0.0, dtype=torch.float32),
        }


class TinyDeepFMRankingDataset(Dataset):
    """Tiny ranking dataset for DeepFM CLI smoke tests."""

    def __init__(self, data_path, split="validation", num_negatives=2):
        self.data_path = data_path
        self.split = split
        self.num_negatives = num_negatives

    def __len__(self) -> int:
        return 2

    def __getitem__(self, index: int):
        num_candidates = self.num_negatives + 1
        sparse_features = torch.tensor(
            [
                [index + 1, candidate + 2, 1, 18, 5, 24, 3]
                for candidate in range(num_candidates)
            ],
            dtype=torch.long,
        )
        labels = torch.zeros(num_candidates, dtype=torch.float32)
        labels[0] = 1.0
        return {
            "dense_features": torch.zeros((num_candidates, 0), dtype=torch.float32),
            "sparse_features": sparse_features,
            "labels": labels,
        }


class DeepFMCliTests(unittest.TestCase):
    """Smoke tests for the DeepFM CLI entrypoint."""

    def test_deepfm_cli_runs_with_tiny_datasets(self) -> None:
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
                        "validation": {"enabled": True, "num_negatives": 2},
                        "paths": {"interactions": "unused_ratings.dat"},
                    }
                ),
                encoding="utf-8",
            )
            model_cfg_path.write_text(
                yaml.safe_dump(
                    {
                        "model": {
                            "dense_input_dim": 0,
                            "embedding_dim": 4,
                            "sparse_fields": {
                                "user_id": {"vocab_size": 64},
                                "item_id": {"vocab_size": 64},
                                "gender": {"vocab_size": 8},
                                "age": {"vocab_size": 128},
                                "occupation": {"vocab_size": 128},
                                "release_year": {"vocab_size": 256},
                                "primary_genre": {"vocab_size": 64},
                            },
                            "deep": {
                                "hidden_dims": [8],
                                "activation": "relu",
                                "dropout": 0.0,
                                "batch_norm": False,
                            },
                            "output": {"activation": "sigmoid"},
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
                        "epochs": 1,
                        "trainer": {
                            "batch_size": 2,
                            "num_workers": 0,
                            "pin_memory": False,
                            "drop_last": False,
                        },
                        "optimizer": {"lr": 1e-3, "weight_decay": 0.0},
                        "scheduler": {"enabled": False},
                        "gradient": {"clip_grad_norm": None},
                        "checkpoint": {
                            "dir": str(tmp_path / "checkpoints"),
                            "best_checkpoint_path": str(
                                tmp_path / "checkpoints" / "best_model.pth"
                            ),
                            "monitor": "val_ndcg@2",
                            "mode": "max",
                        },
                        "early_stop": {"enabled": False, "patience": 0},
                        "logging": {
                            "plot_path": str(
                                tmp_path / "checkpoints" / "training_curves.png"
                            )
                        },
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
                patch.object(deepfm_train, "DeepFMDataset", TinyDeepFMDataset),
                patch.object(
                    deepfm_train, "DeepFMRankingDataset", TinyDeepFMRankingDataset
                ),
                patch.object(
                    deepfm_train,
                    "parse_args",
                    return_value=argparse.Namespace(config=str(experiment_cfg_path)),
                ),
                redirect_stdout(stdout),
            ):
                deepfm_train.main()

            output = stdout.getvalue()
            self.assertIn("Training config |", output)
            self.assertIn("Final test metrics:", output)
            self.assertTrue((tmp_path / "checkpoints" / "best_model.pth").exists())
            self.assertTrue(
                (tmp_path / "checkpoints" / "training_curves_loss.txt").exists()
            )
            self.assertTrue(
                (tmp_path / "checkpoints" / "training_curves_metric.txt").exists()
            )


if __name__ == "__main__":
    unittest.main()
