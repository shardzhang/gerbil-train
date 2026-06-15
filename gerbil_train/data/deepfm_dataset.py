from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset

from gerbil_train.data.shared_bottom_two_tower_dataset import load_ml1m_ratings
from gerbil_train.data.shared_bottom_two_tower_dataset import load_ml1m_movies
from gerbil_train.data.shared_bottom_two_tower_dataset import load_ml1m_users
from gerbil_train.data.shared_bottom_two_tower_dataset import (
    _split_ratings_leave_two_out,
)

__all__ = [
    "DeepFMDataset",
    "DeepFMRankingDataset",
]

SPARSE_FIELD_NAMES = [
    "user_id",
    "item_id",
    "gender",
    "age",
    "occupation",
    "release_year",
    "primary_genre",
]


def _select_ratings_split(ratings: pd.DataFrame, split: str) -> pd.DataFrame:
    """Select one named ratings split for DeepFM."""
    train_ratings, validation_ratings, test_ratings = _split_ratings_leave_two_out(
        ratings
    )
    if split == "full":
        return ratings
    if split == "train":
        return train_ratings
    if split == "validation":
        return validation_ratings
    if split == "test":
        return test_ratings
    raise ValueError(f"Unsupported split: {split}")


def _infer_side_paths(data_path: str | Path) -> tuple[Path, Path]:
    """Infer ``users.dat`` and ``movies.dat`` from the ratings path."""
    base_dir = Path(data_path).resolve().parent
    return base_dir / "users.dat", base_dir / "movies.dat"


def _build_sparse_feature_rows(
    data_path: str | Path,
) -> dict[tuple[int, int], tuple[int, ...]]:
    """Build sparse DeepFM feature rows keyed by ``(user_id, item_id)``."""
    ratings = load_ml1m_ratings(data_path)
    users_path, movies_path = _infer_side_paths(data_path)
    users = load_ml1m_users(users_path)
    movies = load_ml1m_movies(movies_path)

    users_by_id = users.set_index("user_id").to_dict("index")
    movies_by_id = movies.set_index("item_id").to_dict("index")

    max_year = 0
    for movie in movies_by_id.values():
        title = str(movie.get("title", ""))
        if title.endswith(")") and "(" in title:
            maybe_year = title.rsplit("(", 1)[-1].rstrip(")")
            if maybe_year.isdigit():
                max_year = max(max_year, int(maybe_year))

    genre_to_index: dict[str, int] = {}
    feature_rows: dict[tuple[int, int], tuple[int, ...]] = {}
    for row in ratings[["user_id", "item_id"]].itertuples(index=False):
        user_id = int(row.user_id)
        item_id = int(row.item_id)
        user = users_by_id.get(user_id, {})
        movie = movies_by_id.get(item_id, {})

        gender = str(user.get("gender", "U"))
        gender_index = 1 if gender == "M" else 2 if gender == "F" else 0
        age_index = int(user.get("age", 0))
        occupation_index = int(user.get("occupation", 0)) + 1

        title = str(movie.get("title", ""))
        release_year = 0
        if title.endswith(")") and "(" in title:
            maybe_year = title.rsplit("(", 1)[-1].rstrip(")")
            if maybe_year.isdigit():
                release_year = int(maybe_year)
        release_year_index = release_year - 1900 + 1 if release_year > 0 else 0

        primary_genre = str(movie.get("genres", "")).split("|")[0]
        if primary_genre and primary_genre not in genre_to_index:
            genre_to_index[primary_genre] = len(genre_to_index) + 1
        primary_genre_index = genre_to_index.get(primary_genre, 0)

        feature_rows[(user_id, item_id)] = (
            user_id,
            item_id,
            gender_index,
            age_index,
            occupation_index,
            release_year_index,
            primary_genre_index,
        )
    return feature_rows


class DeepFMDataset(Dataset):
    """Pointwise DeepFM dataset built from MovieLens ratings."""

    def __init__(
        self,
        data_path: str | Path,
        split: str = "train",
    ) -> None:
        """Initialize the DeepFM dataset.

        :param data_path: Path to ``ratings.dat``
        :param split: One of ``train``, ``validation``, ``test``, or ``full``
        """
        self.data_path = str(data_path)
        ratings = load_ml1m_ratings(self.data_path)
        self.feature_rows = _build_sparse_feature_rows(self.data_path)
        self.records = [
            (int(row.user_id), int(row.item_id), float(row.rating))
            for row in _select_ratings_split(ratings, split)[
                ["user_id", "item_id", "rating"]
            ].itertuples(index=False)
        ]

    def __len__(self) -> int:
        """Return the dataset size."""
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        """Return one pointwise DeepFM training example."""
        user_id, item_id, rating = self.records[index]
        return {
            "dense_features": torch.zeros(0, dtype=torch.float32),
            "sparse_features": torch.tensor(
                self.feature_rows[(user_id, item_id)],
                dtype=torch.long,
            ),
            "label": torch.tensor(rating / 5.0, dtype=torch.float32),
        }


class DeepFMRankingDataset(Dataset):
    """Ranking dataset for DeepFM validation and test evaluation."""

    def __init__(
        self,
        data_path: str | Path,
        *,
        split: str,
        num_negatives: int = 99,
    ) -> None:
        """Initialize ranking groups for validation or test.

        :param data_path: Path to ``ratings.dat``
        :param split: ``validation`` or ``test``
        :param num_negatives: Number of sampled negatives per query group
        """
        if split not in {"validation", "test"}:
            raise ValueError(
                "DeepFMRankingDataset split must be 'validation' or 'test'"
            )

        self.data_path = str(data_path)
        self.split = split
        self.num_negatives = num_negatives

        ratings = load_ml1m_ratings(self.data_path)
        self.feature_rows = _build_sparse_feature_rows(self.data_path)
        train_ratings, validation_ratings, test_ratings = _split_ratings_leave_two_out(
            ratings
        )

        if split == "validation":
            target_ratings = validation_ratings
            seen_ratings = train_ratings
        else:
            target_ratings = test_ratings
            seen_ratings = pd.concat(
                [train_ratings, validation_ratings], ignore_index=True
            )

        self.records = [
            (int(row.user_id), int(row.item_id), float(row.rating))
            for row in target_ratings[["user_id", "item_id", "rating"]].itertuples(
                index=False
            )
        ]
        self.user_seen_items = {
            int(user_id): {int(item_id) for item_id in item_ids.tolist()}
            for user_id, item_ids in seen_ratings.groupby("user_id")["item_id"]
        }
        self.all_item_ids = sorted(
            int(item_id) for item_id in ratings["item_id"].unique()
        )

    def __len__(self) -> int:
        """Return the number of ranking groups."""
        return len(self.records)

    def _sample_negative_item_ids(
        self,
        user_id: int,
        positive_item_id: int,
        index: int,
    ) -> list[int]:
        """Deterministically sample negative items for one ranking group."""
        seen_items = set(self.user_seen_items.get(user_id, set()))
        negative_item_ids: list[int] = []
        cursor = (index + user_id) % max(len(self.all_item_ids), 1)

        while len(negative_item_ids) < self.num_negatives and self.all_item_ids:
            candidate_item_id = self.all_item_ids[cursor]
            if (
                candidate_item_id not in seen_items
                and candidate_item_id != positive_item_id
            ):
                negative_item_ids.append(candidate_item_id)
            cursor = (cursor + 1) % len(self.all_item_ids)
        return negative_item_ids

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        """Return one ranking group for validation or test."""
        user_id, positive_item_id, rating = self.records[index]
        negative_item_ids = self._sample_negative_item_ids(
            user_id=user_id,
            positive_item_id=positive_item_id,
            index=index,
        )
        candidate_item_ids = [positive_item_id, *negative_item_ids]

        sparse_features = torch.tensor(
            [self.feature_rows[(user_id, item_id)] for item_id in candidate_item_ids],
            dtype=torch.long,
        )
        labels = torch.zeros(len(candidate_item_ids), dtype=torch.float32)
        labels[0] = float(rating)

        return {
            "dense_features": torch.zeros(
                (len(candidate_item_ids), 0), dtype=torch.float32
            ),
            "sparse_features": sparse_features,
            "labels": labels,
        }
