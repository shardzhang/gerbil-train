"""Learning-to-rank dataset helpers and dataloader builders.

Dataset download:
    curl -L "https://hf-mirror.com/datasets/aletovv/MSLRWEB10K/resolve/main/MSLR-WEB10K.pt" -o "MSLR-WEB10K.pt"

Expected ``.pt`` format:
    dict["train" | "vali" | "test"]
    dict[qid: int] -> (
        features: np.ndarray,   # shape = (num_docs_for_query, 136), dtype=float64
        labels: np.ndarray      # shape = (num_docs_for_query,), dtype=int64
    )
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

__all__ = [
    "LearningToRankDataset",
    "build_ltr_dataloaders",
    "load_letor_data",
    "load_mslr_data",
    "load_mslrweb10k_groups",
    "normalize_by_query",
    "normalize_query_features",
]


class LearningToRankDataset(Dataset):
    """Dataset of per-query ranking groups.

    Each sample is a dictionary with:
      - ``qid``: query id
      - ``X``: feature tensor of shape ``[num_docs, num_features]``
      - ``y``: label tensor of shape ``[num_docs]``
    """

    def __init__(self, groups: list[dict[str, Any]]) -> None:
        """Initialize the dataset from preprocessed query groups.

        :param groups: List of query-group dictionaries
        """
        self.groups = groups

    def __len__(self) -> int:
        """Return the number of query groups in the dataset."""
        return len(self.groups)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Return one query group by index.

        :param index: Dataset index
        :return: Query-group dictionary
        """
        return self.groups[index]


def _collate_single_query_group(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate one variable-length query group.

    The current learning-to-rank trainer operates on one query group at a time
    because different queries may contain different numbers of documents.

    :param batch: List of dataset samples produced by the DataLoader
    :return: The single query-group sample in the batch
    """
    if len(batch) != 1:
        raise ValueError(
            "Learning-to-rank DataLoader currently expects batch_size=1 "
            "because query groups have variable document counts."
        )
    return batch[0]


def load_letor_data(file_path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load a LETOR 4.0 text file into arrays.

    :param file_path: Path to a LETOR 4.0 data file
    :return: Tuple of ``(features, labels, qids)`` arrays
    """
    labels = []
    qids = []
    features = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            label = int(parts[0])
            qid = int(parts[1].split(":")[1])

            feat = np.zeros(46)
            for item in parts[2:]:
                if ":" not in item:
                    continue
                fid, val = item.split(":")
                feat[int(fid) - 1] = float(val)

            labels.append(label)
            qids.append(qid)
            features.append(feat)

    return np.array(features), np.array(labels), np.array(qids)


def load_mslr_data(file_path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load an MSLR-WEB10K text file into arrays.

    :param file_path: Path to an MSLR-WEB10K text file
    :return: Tuple of ``(features, labels, qids)`` arrays
    """
    labels = []
    qids = []
    features = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            label = int(parts[0])
            qid = int(parts[1].split(":")[1])

            feat = np.zeros(136)
            for item in parts[2:]:
                if ":" not in item:
                    continue
                fid, val = item.split(":")
                feat[int(fid) - 1] = float(val)

            labels.append(label)
            qids.append(qid)
            features.append(feat)

    return np.array(features), np.array(labels), np.array(qids)


def normalize_by_query(
    features: np.ndarray,
    qids: np.ndarray,
    eps: float = 1e-6,
) -> np.ndarray:
    """Apply z-score normalization independently within each query group.

    :param features: Feature array of shape ``[num_rows, num_features]``
    :param qids: Query ids aligned with ``features``
    :param eps: Small value used to avoid division by zero
    :return: Query-normalized feature array
    """
    unique_qids = np.unique(qids)
    normalized = features.copy()

    for qid in unique_qids:
        mask = qids == qid
        features_for_query = features[mask]

        mean = features_for_query.mean(axis=0, keepdims=True)
        std = features_for_query.std(axis=0, keepdims=True)
        std[std < eps] = 1.0

        normalized[mask] = (features_for_query - mean) / std

    return normalized


def normalize_query_features(features: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Normalize features for a single query group using z-score normalization.

    :param features: 2D array of shape ``[num_docs, num_features]`` for one query group
    :param eps: Small value used to avoid division by zero
    :return: Normalized feature array with the same shape
    """
    mean = features.mean(axis=0, keepdims=True)
    std = features.std(axis=0, keepdims=True)
    std = np.clip(std, eps, None)
    return (features - mean) / std


def load_mslrweb10k_groups(
    file_path: str | Path,
) -> tuple[LearningToRankDataset, LearningToRankDataset, LearningToRankDataset]:
    """Load the MSLR-WEB10K ``.pt`` file and return split datasets.

    :param file_path: Path to the ``MSLR-WEB10K.pt`` file
    :return: Tuple of ``(train_dataset, val_dataset, test_dataset)``
    """
    dataset = torch.load(file_path, map_location="cpu", weights_only=False)

    def process_split(
        split: dict[int, tuple[np.ndarray, np.ndarray]],
    ) -> LearningToRankDataset:
        """Convert one split dictionary into a ranking dataset.

        :param split: Mapping from query id to ``(features, labels)`` arrays
        :return: ``LearningToRankDataset`` for the split
        """
        groups: list[dict[str, Any]] = []
        for qid, (features, labels) in split.items():
            normalized_features = normalize_query_features(
                features.astype(dtype=np.float32, copy=False)
            )
            group_labels = labels.astype(dtype=np.float32, copy=False)
            groups.append(
                {
                    "qid": qid,
                    "X": torch.from_numpy(normalized_features),
                    "y": torch.from_numpy(group_labels),
                }
            )
        return LearningToRankDataset(groups)

    train_dataset = process_split(dataset["train"])
    val_dataset = process_split(dataset["vali"])
    test_dataset = process_split(dataset["test"])
    return train_dataset, val_dataset, test_dataset


def build_ltr_dataloaders(
    file_path: str | Path,
    *,
    train_batch_size: int = 1,
    eval_batch_size: int = 1,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Build train, validation, and test dataloaders for learning-to-rank.

    The current trainer consumes one query group per optimization step, so the
    dataloaders use a custom collate function and require ``batch_size=1``.

    :param file_path: Path to the ``MSLR-WEB10K.pt`` file
    :param train_batch_size: Training batch size; currently must be 1
    :param eval_batch_size: Evaluation batch size; currently must be 1
    :param num_workers: Number of DataLoader worker processes
    :param pin_memory: Whether to enable pinned memory in the dataloaders
    :return: Tuple of ``(train_loader, val_loader, test_loader)``
    """
    if train_batch_size != 1 or eval_batch_size != 1:
        raise ValueError(
            "Learning-to-rank DataLoader currently supports only batch_size=1."
        )

    train_dataset, val_dataset, test_dataset = load_mslrweb10k_groups(file_path)

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=_collate_single_query_group,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=_collate_single_query_group,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=_collate_single_query_group,
    )
    return train_loader, val_loader, test_loader
