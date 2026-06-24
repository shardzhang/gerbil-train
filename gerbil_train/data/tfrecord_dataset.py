"""Abstract TFRecord dataset with overridable target extraction."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from tfrecord import example_pb2
from tfrecord.reader import tfrecord_iterator
from torch.utils.data import IterableDataset, get_worker_info

from gerbil_train.config.model_config import FieldEntry


__all__ = [
    "TFRecordDataset",
    "BatchCollator",
    "FieldEntry",
    "collect_tfrecord_part_files",
    "count_tfrecord_records",
    "load_field_specs",
    "load_field_stats",
    "load_target_size",
    "BinaryTFRecordDataset",
    "MultiTFRecordDataset",
]


def load_field_specs(path: str | Path) -> list[FieldEntry]:
    """Load field specs from ``pos_map.txt`` (columns: ``name,index,type,dim``)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Field spec file not found: {path}")
    specs: list[FieldEntry] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("field_name,"):
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 4:
                continue
            specs.append(FieldEntry(field_name=parts[0], field_index=int(parts[1]), field_type=int(parts[2]), dim=int(parts[3])))
    specs.sort(key=lambda s: s.field_index)
    return specs


def load_target_size(path: str | Path) -> int:
    """Load target vocabulary size from ``pos_map.json``."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Target size file not found: {path}")
    
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    
    if "target_size" in payload:
        return int(payload["target_size"])
    
    targets = payload.get("targets", {})
    if not isinstance(targets, Mapping) or not targets:
        raise ValueError("Unable to infer target_size from JSON")
    return len(targets)


def load_field_stats(path: str | Path) -> dict[str, dict[int, tuple[float, float]]]:
    """Load per-position mean/std for continuous features from ``pos_map.json``."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Field stats file not found: {path}")
    
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    # dict[field_name, dict[pos, (mean, std)]]    
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


def collect_tfrecord_part_files(root_dir: str | Path) -> list[Path]:
    """Collect ``part-r-*`` TFRecord files under a root directory."""
    root = Path(root_dir)
    if not root.exists():
        raise FileNotFoundError(f"TFRecord root directory not found: {root}")
    
    files = [
        p for p in root.glob("part-r-*") if p.is_file()
        and not p.name.startswith(".") and not p.name.endswith(".crc")
    ]
    files = sorted(files)
    if not files:
        raise FileNotFoundError(f"No TFRecord part files found in {root}")
    return files


def count_tfrecord_records(files: Sequence[str | Path]) -> int:
    """Count total TFRecord records without parsing protobuf."""
    total = 0
    for f in files:
        it = tfrecord_iterator(data_path=str(Path(f).resolve()), index_path=None)
        for _ in it:
            total += 1
    return total


class BatchCollator:
    """Collate function that packs sparse per-field features for EmbeddingBag."""
    def __init__(self, field_names: Sequence[str]) -> None:
        self.field_names = list(field_names)

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        targets: torch.Tensor = torch.tensor([int(sample["targets"]) for sample in batch], dtype=torch.long)

        # dict[field_name, dict[indices, offsets, weights]]
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


class TFRecordDataset(IterableDataset):
    """Abstract TFRecord dataset with overridable target extraction.

    Subclasses must implement :meth:`_extract_target`.
    """
    def __init__(
        self,
        tfrecord_files: Sequence[str | Path],
        field_specs: Sequence[FieldEntry],
        *,
        field_stats: dict[str, dict[int, tuple[float, float]]] | None = None,
        shuffle_files: bool = False,
        shuffle_buffer: int = 0,
        seed: int = 42,
    ) -> None:
        self.tfrecord_files = [str(Path(path).resolve()) for path in tfrecord_files]
        self.field_specs = list(field_specs)
        self.field_stats = field_stats
        self.shuffle_files = shuffle_files
        self.shuffle_buffer = shuffle_buffer
        self.seed = seed

        if not self.tfrecord_files:
            raise ValueError("tfrecord_files must not be empty")
        if not self.field_specs:
            raise ValueError("field_specs must not be empty")


    def _extract_target(self, example: example_pb2.Example) -> int | float:
        """Extract the target label from a TFRecord ``Example``.
        Subclasses must override this method.
        """
        raise NotImplementedError


    def _extract_field_values(self, features: dict[str, example_pb2.Feature], field_name: str) -> tuple[list[int], list[float]]:
        """Extract indices and values for a named field from a TFRecord example.
        :return: Tuple of ``(indices, values)``.
        """
        idx_feat = features.get(f"{field_name}_index")
        val_feat = features.get(f"{field_name}_value")
        if idx_feat is None or val_feat is None:
            return [], []
        indices: list[int] = [int(v) for v in idx_feat.int64_list.value]
        values: list[float] = [float(v) for v in val_feat.float_list.value]
        return indices, values


    def _select_files_for_worker(self) -> list[str]:
        """Select files for the current worker."""  
        worker = get_worker_info()
        if worker is None:
            selected = list(self.tfrecord_files)
            worker_seed = self.seed
        else:
            selected = self.tfrecord_files[worker.id :: worker.num_workers]
            worker_seed = self.seed + worker.id
        if self.shuffle_files:
            random.Random(worker_seed).shuffle(selected)
        return selected


    def __iter__(self):
        """Iterator over the dataset.
        Note: key of sample is field_name
        """
        selected_files = self._select_files_for_worker()
        buf: list[dict[str, Any]] = []
        for file_path in selected_files:
            iterator = tfrecord_iterator(data_path=file_path, index_path=None)
            for raw_record in iterator:
                example: example_pb2.Example = example_pb2.Example()
                example.ParseFromString(raw_record)
                target = self._extract_target(example)
                features: dict[str, example_pb2.Feature] = example.features.feature
                field_indices: dict[str, list[int]] = {}
                field_values: dict[str, list[float]] = {}
                for f_spec in self.field_specs:
                    indices, values = self._extract_field_values(features, f_spec.field_name)
                    if f_spec.field_type == 0 and self.field_stats is not None:
                        pos_stats = self.field_stats.get(f_spec.field_name)
                        if pos_stats:
                            values = [
                                (v - pos_stats.get(idx, (0.0, 1.0))[0]) / pos_stats.get(idx, (0.0, 1.0))[1]
                                for idx, v in zip(indices, values)
                            ]
                    field_indices[f_spec.field_name] = indices
                    field_values[f_spec.field_name] = values

                sample = {
                    "targets": target, 
                    "field_indices": field_indices, 
                    "field_values": field_values
                }
                if self.shuffle_buffer > 0:
                    buf.append(sample)
                    if len(buf) >= self.shuffle_buffer:
                        random.shuffle(buf)
                        yield from buf
                        buf = []
                else:
                    yield sample

        if self.shuffle_buffer > 0 and buf:
            random.shuffle(buf)
            yield from buf


class BinaryTFRecordDataset(TFRecordDataset):
    """TFRecord dataset for binary CTR classification."""
    def _extract_target(self, example: example_pb2.Example) -> float:
        target_feature = example.features.feature.get("target")
        if target_feature is None:
            raise ValueError("Missing target feature in TFRecord example")
        if target_feature.float_list.value:
            rating = float(target_feature.float_list.value[0])
            return 1.0 if rating > 3.0 else 0.0
        raise ValueError("Target feature exists but has no values")


class MultiTFRecordDataset(TFRecordDataset):
    """TFRecord dataset for multi-class classification."""
    def _extract_target(self, example: example_pb2.Example) -> int:
        target_feature: example_pb2.Feature | None = example.features.feature.get("target")
        if target_feature is None:
            raise ValueError("Missing target feature in TFRecord example")
        if target_feature.float_list.value:
            return int(target_feature.float_list.value[0])
        raise ValueError("Target feature exists but has no values")
