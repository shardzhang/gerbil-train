from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import Dataset


class SharedBottomTwoTowerImplicitDataset(Dataset):
    """Dataset for implicit pretraining of SharedBottomTwoTower.

    Expected batch item format:
      {
        "query_features": Tensor[Dq],
        "pos_item_features": Tensor[Di],
        "neg_item_features": Tensor[N, Di],
      }
    """

    def __init__(
        self,
        data_path: str | Path,
        query_input_dim: int,
        item_input_dim: int,
        num_negatives: int,
        size: int = 10000,
    ) -> None:
        """
        :param data_path: Path to the dataset file (not used in this dummy implementation)
        :param query_input_dim: Dimensionality of query features
        :param item_input_dim: Dimensionality of item features
        :param num_negatives: Number of negative samples per positive sample
        :param size: Number of samples in the dataset (for this dummy implementation)
        """
        self.data_path = str(data_path)
        self.query_input_dim = query_input_dim
        self.item_input_dim = item_input_dim
        self.num_negatives = num_negatives
        self.size = size

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        query_features = torch.randn(self.query_input_dim)
        pos_item_features = torch.randn(self.item_input_dim)
        neg_item_features = torch.randn(self.num_negatives, self.item_input_dim)

        return {
            "query_features": query_features,
            "pos_item_features": pos_item_features,
            "neg_item_features": neg_item_features,
        }


class SharedBottomTwoTowerExplicitDataset(Dataset):
    """Dataset for explicit fine-tuning of SharedBottomTwoTower.

    Expected batch item format:
      {
        "query_features": Tensor[Dq],
        "item_features": Tensor[Di],
        "label": Tensor[],
      }
    """

    def __init__(
        self,
        data_path: str | Path,
        query_input_dim: int,
        item_input_dim: int,
        size: int = 5000,
    ) -> None:
        """
        :param data_path: Path to the dataset file (not used in this dummy implementation)
        :param query_input_dim: Dimensionality of query features
        :param item_input_dim: Dimensionality of item features
        :param size: Number of samples in the dataset (for this dummy implementation)
        """
        self.data_path = str(data_path)
        self.query_input_dim = query_input_dim
        self.item_input_dim = item_input_dim
        self.size = size

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        query_features = torch.randn(self.query_input_dim)
        item_features = torch.randn(self.item_input_dim)
        label = torch.rand(())

        return {
            "query_features": query_features,
            "item_features": item_features,
            "label": label,
        }