from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from tfrecord import example_pb2
from tfrecord.reader import tfrecord_iterator
from torch.utils.data import IterableDataset, get_worker_info

__all__ = [
    "GwENBatchCollator",
    "GwENFieldSpec",
    "GwENTFRecordDataset",
    "collect_tfrecord_part_files",
    "load_gwen_field_specs",
    "load_gwen_field_stats",
    "load_target_size",
]


@dataclass(frozen=True)
class GwENFieldSpec:
    """One GwEN feature-field specification from ``nn_pos_map.txt``."""

    name: str
    index: int
    field_type: int
    dim: int


def load_gwen_field_specs(path: str | Path) -> list[GwENFieldSpec]:
    """Load field specs from ``nn_pos_map.txt``.

    Expected format:
        ``field_name,field_index,field_type,dim``
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"nn_pos_map.txt not found: {path}")

    specs: list[GwENFieldSpec] = []
    with path.open("r", encoding="utf-8") as f:
        for line_number, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("field_name,"):
                continue
            parts = [part.strip() for part in line.split(",")]
            if len(parts) < 4:
                raise ValueError(f"Invalid nn_pos_map line at {path}:{line_number}")
            specs.append(GwENFieldSpec(name=parts[0], index=int(parts[1]), field_type=int(parts[2]), dim=int(parts[3])))
    specs.sort(key=lambda spec: spec.index)
    return specs


def load_target_size(path: str | Path) -> int:
    """Load target vocabulary size from ``nn_pos_map.json``."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"nn_pos_map.json not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    if "target_size" in payload:
        return int(payload["target_size"])

    targets = payload.get("targets", {})
    if not isinstance(targets, Mapping) or not targets:
        raise ValueError("Unable to infer target_size from nn_pos_map.json")

    return len(targets)


def collect_tfrecord_part_files(root_dir: str | Path) -> list[Path]:
    """Collect TFRecord part files under one root directory."""
    root = Path(root_dir)
    if not root.exists():
        raise FileNotFoundError(f"TFRecord root directory not found: {root}")

    files = [
        path for path in root.glob("part-r-*") if path.is_file()
        and not path.name.startswith(".")
        and not path.name.endswith(".crc")
    ]
    files = sorted(files)
    # print(f"root_dir: {root_dir}")
    # for i, file in enumerate(files):
    #     print(f"{i}: {file}")
    if not files:
        raise FileNotFoundError(f"No TFRecord part files found in {root}")
    return files


def load_gwen_field_stats(path: str | Path) -> dict[str, dict[int, tuple[float, float]]]:
    """Load continuous-feature mean and std from ``pos_map.json``.

    Returns a mapping ``field_name -> {pos: (mean, std)}`` for field_type=0 features.
    Each bucket position has its own mean/std.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"pos_map.json not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    stats: dict[str, dict[int, tuple[float, float]]] = {}
    for feature in payload.get("features", []):
        if int(feature.get("field_type", 1)) != 0:
            continue
        per_pos: dict[int, tuple[float, float]] = {}
        for entry in feature.get("entries", []):
            pos = int(entry.get("pos", 0))
            if entry.get("hash") is not None:
                per_pos[pos] = (float(entry["mean"]), float(entry["std"]))
        if per_pos:
            stats[feature["field_name"]] = per_pos
    return stats


class GwENTFRecordDataset(IterableDataset):
    """Iterable TFRecord dataset for GwEN multi-class training."""

    def __init__(
        self,
        tfrecord_files: Sequence[str | Path],
        field_names: Sequence[str],
        *,
        field_stats: dict[str, dict[int, tuple[float, float]]] | None = None,
        shuffle_files: bool = False,
        seed: int = 42,
    ) -> None:
        self.tfrecord_files = [str(Path(path).resolve()) for path in tfrecord_files]
        self.field_names = list(field_names)
        self.field_stats = field_stats
        self.shuffle_files = shuffle_files
        self.seed = seed

        if not self.tfrecord_files:
            raise ValueError("tfrecord_files must not be empty")
        if not self.field_names:
            raise ValueError("field_names must not be empty")

    def _select_files_for_worker(self) -> list[str]:
        worker = get_worker_info()
        if worker is None:
            selected_files = list(self.tfrecord_files)
            worker_seed = self.seed
        else:
            selected_files = self.tfrecord_files[worker.id :: worker.num_workers]
            worker_seed = self.seed + worker.id

        if self.shuffle_files:
            random.Random(worker_seed).shuffle(selected_files)
        return selected_files

    @staticmethod
    def _extract_target(example: example_pb2.Example) -> int:
        target_feature = example.features.feature.get("target")
        if target_feature is None:
            raise ValueError("Missing target feature in TFRecord example")

        if target_feature.float_list.value:
            return int(target_feature.float_list.value[0])
        
        raise ValueError("Target feature exists but has no values")

    def _extract_field_values(self, example: example_pb2.Example, field_name: str) -> tuple[list[int], list[float]]:
        features = example.features.feature
        index_feature = features.get(f"{field_name}_index")
        value_feature = features.get(f"{field_name}_value")

        if index_feature is None or value_feature is None:
            return [], []

        index_feature = [int(value) for value in index_feature.int64_list.value]
        value_feature = [float(value) for value in value_feature.float_list.value]
        return index_feature, value_feature

    def __iter__(self):
        selected_files = self._select_files_for_worker()
        for file_path in selected_files:
            iterator = tfrecord_iterator(data_path=file_path, index_path=None)
            for raw_record in iterator:
                example = example_pb2.Example()
                example.ParseFromString(raw_record)

                target = self._extract_target(example)
                field_indices: dict[str, list[int]] = {}
                field_values: dict[str, list[float]] = {}
                for field_name in self.field_names:
                    indices, values = self._extract_field_values(example, field_name)
                    if self.field_stats is not None and field_name in self.field_stats:
                        pos_stats = self.field_stats[field_name]
                        values = [
                            (v - pos_stats.get(idx, (0.0, 1.0))[0]) / pos_stats.get(idx, (0.0, 1.0))[1]
                            for idx, v in zip(indices, values)
                        ]
                    field_indices[field_name] = indices
                    field_values[field_name] = values

                yield {
                    "targets": target,
                    "field_indices": field_indices,
                    "field_values": field_values,
                }


class GwENBatchCollator:
    """Collate function that packs sparse per-field features for EmbeddingBag."""

    def __init__(self, field_names: Sequence[str]) -> None:
        self.field_names = list(field_names)

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        targets = torch.tensor([int(sample["targets"]) for sample in batch], dtype=torch.long)

        feature_bags: dict[str, dict[str, torch.Tensor]] = {}
        for field_name in self.field_names:
            offsets: list[int] = []
            indices: list[int] = []
            weights: list[float] = []
            cursor = 0
            for sample in batch:
                offsets.append(cursor)
                field_indices = sample["field_indices"][field_name]
                field_values = sample["field_values"][field_name]
                indices.extend(field_indices)
                weights.extend(field_values)
                cursor += len(field_indices)

            feature_bags[field_name] = {
                "indices": torch.tensor(indices, dtype=torch.long),
                "offsets": torch.tensor(offsets, dtype=torch.long),
                "weights": torch.tensor(weights, dtype=torch.float32),
            }

        return {
            "targets": targets,
            "feature_bags": feature_bags,
        }
