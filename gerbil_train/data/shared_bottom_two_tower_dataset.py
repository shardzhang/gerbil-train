from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset


__all__ = [
    "ExplicitDataset",
    "ImplicitDataset",
    "RankingValidationDataset",
    "load_ml1m_movies",
    "load_ml1m_ratings",
    "load_ml1m_users",
]


def load_ml1m_ratings(
    data_path: str | Path,
    *,
    separator: str = "::",
    encoding: str = "latin-1",
) -> pd.DataFrame:
    """Load MovieLens 1M ``ratings.dat`` into a dataframe.

    The returned dataframe uses normalized snake-case column names that match
    the rest of the project configuration.

    :param data_path: Path to ``ratings.dat``
    :param separator: Field separator used by the raw file
    :param encoding: File encoding
    :return: Parsed ratings dataframe
    """
    ratings = pd.read_csv(
        data_path,
        sep=separator,
        engine="python",
        header=None,
        names=["user_id", "item_id", "rating", "timestamp"],
        encoding=encoding,
    )
    return ratings.astype(
        {
            "user_id": "int64",
            "item_id": "int64",
            "rating": "float32",
            "timestamp": "int64",
        }
    )


def load_ml1m_users(
    data_path: str | Path,
    *,
    separator: str = "::",
    encoding: str = "latin-1",
) -> pd.DataFrame:
    """Load MovieLens 1M ``users.dat`` into a dataframe."""
    users = pd.read_csv(
        data_path,
        sep=separator,
        engine="python",
        header=None,
        names=["user_id", "gender", "age", "occupation", "zip_code"],
        encoding=encoding,
    )
    return users.astype(
        {
            "user_id": "int64",
            "gender": "string",
            "age": "int64",
            "occupation": "int64",
            "zip_code": "string",
        }
    )


def load_ml1m_movies(
    data_path: str | Path,
    *,
    separator: str = "::",
    encoding: str = "latin-1",
) -> pd.DataFrame:
    """Load MovieLens 1M ``movies.dat`` into a dataframe."""
    movies = pd.read_csv(
        data_path,
        sep=separator,
        engine="python",
        header=None,
        names=["item_id", "title", "genres"],
        encoding=encoding,
    )
    return movies.astype(
        {
            "item_id": "int64",
            "title": "string",
            "genres": "string",
        }
    )


def _infer_ml1m_side_paths(ratings_path: str | Path) -> tuple[Path | None, Path | None]:
    """Infer ``users.dat`` and ``movies.dat`` paths next to ``ratings.dat``."""
    base_dir = Path(ratings_path).resolve().parent
    users_path = base_dir / "users.dat"
    movies_path = base_dir / "movies.dat"
    return (
        users_path if users_path.exists() else None,
        movies_path if movies_path.exists() else None,
    )


def _stable_hash_index(token: str, dim: int, salt: str) -> int:
    """Map one token to a stable feature index."""
    digest = hashlib.blake2b(
        f"{salt}:{token}".encode("utf-8"),
        digest_size=8,
    ).digest()
    return int.from_bytes(digest, "little") % dim


def _build_dense_feature_vector(
    *,
    dim: int,
    numeric_values: list[float],
    tokens: list[str],
) -> Tensor:
    """Build a deterministic dense feature vector from numeric values and tokens."""
    feature_vector = torch.zeros(dim, dtype=torch.float32)
    numeric_dim = min(len(numeric_values), dim)

    for index in range(numeric_dim):
        feature_vector[index] = float(numeric_values[index])

    hashed_dim = dim - numeric_dim
    if hashed_dim > 0:
        offset = numeric_dim
        for token in tokens:
            primary_index = _stable_hash_index(token, hashed_dim, salt="primary")
            secondary_index = _stable_hash_index(token, hashed_dim, salt="secondary")
            feature_vector[offset + primary_index] += 1.0
            feature_vector[offset + secondary_index] += 0.5

    norm = torch.linalg.vector_norm(feature_vector)
    if norm > 0:
        feature_vector = feature_vector / norm
    return feature_vector


def _extract_movie_year(title: str) -> int | None:
    """Extract a release year from a MovieLens title string."""
    match = re.search(r"\((\d{4})\)\s*$", title)
    if match is None:
        return None
    return int(match.group(1))


def _build_user_feature_map(
    ratings: pd.DataFrame,
    users: pd.DataFrame | None,
    query_input_dim: int,
) -> dict[int, Tensor]:
    """Build deterministic user feature tensors keyed by user id."""
    user_ids = sorted(int(user_id) for user_id in ratings["user_id"].unique())
    user_activity = ratings.groupby("user_id").size().to_dict()
    max_user_id = max(user_ids, default=1)
    max_activity = max((int(count) for count in user_activity.values()), default=1)

    users_by_id: dict[int, dict[str, object]] = {}
    if users is not None:
        users_by_id = users.set_index("user_id").to_dict("index")

    max_age = max(
        (int(row.get("age", 0)) for row in users_by_id.values()),
        default=1,
    )
    max_occupation = max(
        (int(row.get("occupation", 0)) for row in users_by_id.values()),
        default=1,
    )

    feature_map: dict[int, Tensor] = {}
    for user_id in user_ids:
        user_row = users_by_id.get(user_id, {})
        gender = str(user_row.get("gender", "U"))
        age = int(user_row.get("age", 0))
        occupation = int(user_row.get("occupation", 0))
        zip_code = str(user_row.get("zip_code", "00000"))
        numeric_values = [
            user_id / max_user_id,
            int(user_activity.get(user_id, 0)) / max_activity,
            1.0 if gender == "M" else 0.0,
            age / max_age if max_age > 0 else 0.0,
            occupation / max_occupation if max_occupation > 0 else 0.0,
        ]
        tokens = [
            f"user_id={user_id}",
            f"gender={gender}",
            f"age={age}",
            f"occupation={occupation}",
            f"zip_prefix={zip_code[:3]}",
        ]
        feature_map[user_id] = _build_dense_feature_vector(
            dim=query_input_dim,
            numeric_values=numeric_values,
            tokens=tokens,
        )
    return feature_map


def _build_item_feature_map(
    ratings: pd.DataFrame,
    movies: pd.DataFrame | None,
    item_input_dim: int,
) -> dict[int, Tensor]:
    """Build deterministic item feature tensors keyed by movie id."""
    item_ids = sorted(int(item_id) for item_id in ratings["item_id"].unique())
    item_popularity = ratings.groupby("item_id").size().to_dict()
    max_item_id = max(item_ids, default=1)
    max_popularity = max((int(count) for count in item_popularity.values()), default=1)

    movies_by_id: dict[int, dict[str, object]] = {}
    if movies is not None:
        movies_by_id = movies.set_index("item_id").to_dict("index")

    feature_map: dict[int, Tensor] = {}
    for item_id in item_ids:
        movie_row = movies_by_id.get(item_id, {})
        title = str(movie_row.get("title", ""))
        year = _extract_movie_year(title)
        genres = str(movie_row.get("genres", "")).split("|")
        numeric_values = [
            item_id / max_item_id,
            int(item_popularity.get(item_id, 0)) / max_popularity,
            (year or 0) / 2100.0,
        ]
        tokens = [f"item_id={item_id}"]
        if year is not None:
            tokens.append(f"year={year}")
        tokens.extend(f"genre={genre}" for genre in genres if genre)
        feature_map[item_id] = _build_dense_feature_vector(
            dim=item_input_dim,
            numeric_values=numeric_values,
            tokens=tokens,
        )
    return feature_map


def _load_ml1m_feature_maps(
    data_path: str | Path,
    query_input_dim: int,
    item_input_dim: int,
) -> tuple[pd.DataFrame, dict[int, Tensor], dict[int, Tensor]]:
    """Load MovieLens artifacts and build user/item feature maps."""
    ratings = load_ml1m_ratings(data_path)
    users_path, movies_path = _infer_ml1m_side_paths(data_path)
    users = load_ml1m_users(users_path) if users_path is not None else None
    movies = load_ml1m_movies(movies_path) if movies_path is not None else None
    query_feature_map = _build_user_feature_map(ratings, users, query_input_dim)
    item_feature_map = _build_item_feature_map(ratings, movies, item_input_dim)
    return ratings, query_feature_map, item_feature_map


def _split_ratings_leave_two_out(
    ratings: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split ratings into train, validation, and test parts using leave-two-out."""
    sorted_ratings = ratings.sort_values(["user_id", "timestamp", "item_id"])
    train_parts: list[pd.DataFrame] = []
    validation_parts: list[pd.DataFrame] = []
    test_parts: list[pd.DataFrame] = []

    for _, user_group in sorted_ratings.groupby("user_id", sort=False):
        if len(user_group) <= 2:
            train_parts.append(user_group)
            continue
        train_parts.append(user_group.iloc[:-2])
        validation_parts.append(user_group.iloc[-2:-1])
        test_parts.append(user_group.iloc[-1:])

    train_ratings = (
        pd.concat(train_parts, ignore_index=True)
        if train_parts
        else sorted_ratings.iloc[0:0].copy()
    )
    validation_ratings = (
        pd.concat(validation_parts, ignore_index=True)
        if validation_parts
        else sorted_ratings.iloc[0:0].copy()
    )
    test_ratings = (
        pd.concat(test_parts, ignore_index=True)
        if test_parts
        else sorted_ratings.iloc[0:0].copy()
    )
    return train_ratings, validation_ratings, test_ratings


def _select_ratings_split(ratings: pd.DataFrame, split: str) -> pd.DataFrame:
    """Select one named ratings split."""
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


class ImplicitDataset(Dataset):
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
        split: str = "full",
    ) -> None:
        """
        :param data_path: Path to the dataset file (not used in this dummy implementation)
        :param query_input_dim: Dimensionality of query features
        :param item_input_dim: Dimensionality of item features
        :param num_negatives: Number of negative samples per positive sample
        :param size: Unused compatibility parameter
        :param split: Ratings split name (``full`` or ``train``)
        """
        self.data_path = str(data_path)
        self.query_input_dim = query_input_dim
        self.item_input_dim = item_input_dim
        self.num_negatives = num_negatives
        self.size = size
        ratings, self.query_feature_map, self.item_feature_map = (
            _load_ml1m_feature_maps(
                data_path=self.data_path,
                query_input_dim=query_input_dim,
                item_input_dim=item_input_dim,
            )
        )
        selected_ratings = _select_ratings_split(ratings, split)
        self.interactions = [
            (int(row.user_id), int(row.item_id))
            for row in selected_ratings[["user_id", "item_id"]].itertuples(index=False)
        ]
        self.user_positive_items = {
            int(user_id): {int(item_id) for item_id in item_ids.tolist()}
            for user_id, item_ids in selected_ratings.groupby("user_id")["item_id"]
        }
        self.all_item_ids = sorted(self.item_feature_map)

        if not self.all_item_ids:
            raise ValueError("The implicit dataset requires at least one item.")

    def __len__(self) -> int:
        """Return the number of observed positive interactions."""
        return len(self.interactions)

    def _sample_negative_item_ids(self, user_id: int, index: int) -> list[int]:
        """Deterministically sample negative items for one user."""
        positive_items = self.user_positive_items[user_id]
        if len(positive_items) >= len(self.all_item_ids):
            raise ValueError("Negative sampling requires at least one unseen item.")

        negatives: list[int] = []
        cursor = (index + user_id) % len(self.all_item_ids)
        while len(negatives) < self.num_negatives:
            candidate_item_id = self.all_item_ids[cursor]
            if candidate_item_id not in positive_items:
                negatives.append(candidate_item_id)
            cursor = (cursor + 1) % len(self.all_item_ids)
        return negatives

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        """Return one implicit-training sample built from real interactions.

        :param index: Sample index
        :return: Dictionary containing implicit-training tensors
        """
        user_id, pos_item_id = self.interactions[index]
        negative_item_ids = self._sample_negative_item_ids(user_id, index)

        query_features = self.query_feature_map[user_id].clone()
        pos_item_features = self.item_feature_map[pos_item_id].clone()
        neg_item_features = torch.stack(
            [self.item_feature_map[item_id] for item_id in negative_item_ids],
            dim=0,
        )

        return {
            "query_features": query_features,
            "pos_item_features": pos_item_features,
            "neg_item_features": neg_item_features,
        }


class ExplicitDataset(Dataset):
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
        split: str = "full",
    ) -> None:
        """
        :param data_path: Path to the dataset file (not used in this dummy implementation)
        :param query_input_dim: Dimensionality of query features
        :param item_input_dim: Dimensionality of item features
        :param size: Unused compatibility parameter
        :param split: Ratings split name (``full`` or ``train``)
        """
        self.data_path = str(data_path)
        self.query_input_dim = query_input_dim
        self.item_input_dim = item_input_dim
        self.size = size
        ratings, self.query_feature_map, self.item_feature_map = (
            _load_ml1m_feature_maps(
                data_path=self.data_path,
                query_input_dim=query_input_dim,
                item_input_dim=item_input_dim,
            )
        )
        selected_ratings = _select_ratings_split(ratings, split)
        self.records = [
            (int(row.user_id), int(row.item_id), float(row.rating))
            for row in selected_ratings[["user_id", "item_id", "rating"]].itertuples(
                index=False
            )
        ]

    def __len__(self) -> int:
        """Return the number of observed explicit ratings."""
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        """Return one explicit-training sample built from real ratings.

        :param index: Sample index
        :return: Dictionary containing explicit-training tensors
        """
        user_id, item_id, rating = self.records[index]
        query_features = self.query_feature_map[user_id].clone()
        item_features = self.item_feature_map[item_id].clone()
        label = torch.tensor(rating, dtype=torch.float32)

        return {
            "query_features": query_features,
            "item_features": item_features,
            "label": label,
        }


class RankingValidationDataset(Dataset):
    """Validation dataset for ranking metrics with leave-one-out positives."""

    def __init__(
        self,
        data_path: str | Path,
        query_input_dim: int,
        item_input_dim: int,
        num_negatives: int = 99,
    ) -> None:
        """Initialize ranking validation groups built from raw ml-1m data.

        :param data_path: Path to ``ratings.dat``
        :param query_input_dim: Dimensionality of query features
        :param item_input_dim: Dimensionality of item features
        :param num_negatives: Number of sampled negatives per validation query
        """
        self.data_path = str(data_path)
        self.query_input_dim = query_input_dim
        self.item_input_dim = item_input_dim
        self.num_negatives = num_negatives

        ratings, self.query_feature_map, self.item_feature_map = (
            _load_ml1m_feature_maps(
                data_path=self.data_path,
                query_input_dim=query_input_dim,
                item_input_dim=item_input_dim,
            )
        )
        train_ratings, validation_ratings, _ = _split_ratings_leave_two_out(ratings)
        self.validation_records = [
            (int(row.user_id), int(row.item_id), float(row.rating))
            for row in validation_ratings[["user_id", "item_id", "rating"]].itertuples(
                index=False
            )
        ]
        self.train_positive_items = {
            int(user_id): {int(item_id) for item_id in item_ids.tolist()}
            for user_id, item_ids in train_ratings.groupby("user_id")["item_id"]
        }
        self.all_item_ids = sorted(self.item_feature_map)

    def __len__(self) -> int:
        """Return the number of validation query groups."""
        return len(self.validation_records)

    def _sample_negative_item_ids(
        self,
        user_id: int,
        positive_item_id: int,
        index: int,
    ) -> list[int]:
        """Deterministically sample negative validation items for one user."""
        positive_items = set(self.train_positive_items.get(user_id, set()))
        negative_item_ids: list[int] = []
        cursor = (index + user_id) % max(len(self.all_item_ids), 1)

        while len(negative_item_ids) < self.num_negatives and self.all_item_ids:
            candidate_item_id = self.all_item_ids[cursor]
            if (
                candidate_item_id not in positive_items
                and candidate_item_id != positive_item_id
            ):
                negative_item_ids.append(candidate_item_id)
            cursor = (cursor + 1) % len(self.all_item_ids)
        return negative_item_ids

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        """Return one validation ranking group."""
        user_id, positive_item_id, rating = self.validation_records[index]
        negative_item_ids = self._sample_negative_item_ids(
            user_id=user_id,
            positive_item_id=positive_item_id,
            index=index,
        )
        candidate_item_ids = [positive_item_id, *negative_item_ids]
        num_candidates = len(candidate_item_ids)

        query_features = self.query_feature_map[user_id].repeat(num_candidates, 1)
        item_features = torch.stack(
            [self.item_feature_map[item_id] for item_id in candidate_item_ids],
            dim=0,
        )
        labels = torch.zeros(num_candidates, dtype=torch.float32)
        labels[0] = float(rating)

        return {
            "query_features": query_features,
            "item_features": item_features,
            "labels": labels,
        }


class RankingTestDataset(Dataset):
    """Test dataset for ranking metrics with leave-two-out positives."""

    def __init__(
        self,
        data_path: str | Path,
        query_input_dim: int,
        item_input_dim: int,
        num_negatives: int = 99,
    ) -> None:
        """Initialize ranking test groups built from raw ml-1m data."""
        self.data_path = str(data_path)
        self.query_input_dim = query_input_dim
        self.item_input_dim = item_input_dim
        self.num_negatives = num_negatives

        ratings, self.query_feature_map, self.item_feature_map = (
            _load_ml1m_feature_maps(
                data_path=self.data_path,
                query_input_dim=query_input_dim,
                item_input_dim=item_input_dim,
            )
        )
        train_ratings, validation_ratings, test_ratings = _split_ratings_leave_two_out(
            ratings
        )
        seen_ratings = pd.concat([train_ratings, validation_ratings], ignore_index=True)
        self.test_records = [
            (int(row.user_id), int(row.item_id), float(row.rating))
            for row in test_ratings[["user_id", "item_id", "rating"]].itertuples(
                index=False
            )
        ]
        self.seen_positive_items = {
            int(user_id): {int(item_id) for item_id in item_ids.tolist()}
            for user_id, item_ids in seen_ratings.groupby("user_id")["item_id"]
        }
        self.all_item_ids = sorted(self.item_feature_map)

    def __len__(self) -> int:
        """Return the number of test query groups."""
        return len(self.test_records)

    def _sample_negative_item_ids(
        self,
        user_id: int,
        positive_item_id: int,
        index: int,
    ) -> list[int]:
        """Deterministically sample negative test items for one user."""
        positive_items = set(self.seen_positive_items.get(user_id, set()))
        negative_item_ids: list[int] = []
        cursor = (index + user_id) % max(len(self.all_item_ids), 1)

        while len(negative_item_ids) < self.num_negatives and self.all_item_ids:
            candidate_item_id = self.all_item_ids[cursor]
            if (
                candidate_item_id not in positive_items
                and candidate_item_id != positive_item_id
            ):
                negative_item_ids.append(candidate_item_id)
            cursor = (cursor + 1) % len(self.all_item_ids)
        return negative_item_ids

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        """Return one test ranking group."""
        user_id, positive_item_id, rating = self.test_records[index]
        negative_item_ids = self._sample_negative_item_ids(
            user_id=user_id,
            positive_item_id=positive_item_id,
            index=index,
        )
        candidate_item_ids = [positive_item_id, *negative_item_ids]
        num_candidates = len(candidate_item_ids)

        query_features = self.query_feature_map[user_id].repeat(num_candidates, 1)
        item_features = torch.stack(
            [self.item_feature_map[item_id] for item_id in candidate_item_ids],
            dim=0,
        )
        labels = torch.zeros(num_candidates, dtype=torch.float32)
        labels[0] = float(rating)

        return {
            "query_features": query_features,
            "item_features": item_features,
            "labels": labels,
        }
