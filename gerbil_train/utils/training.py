"""Shared utilities for training CLI scripts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from torch.utils.data import DataLoader

from gerbil_train.config.model_config import FieldEntry, load_enabled_field_entries
from gerbil_train.data.tfrecord_dataset import (
    BatchCollator, BinaryTFRecordDataset,
    collect_tfrecord_part_files, load_field_stats, load_target_size,
)


def build_loader(
    files: list[Path],
    field_entries: list[FieldEntry],
    *,
    batch_size: int,
    shuffle_files: bool,
    shuffle_buffer: int | None = None,
    num_workers: int = 0,
    pin_memory: bool = False,
    drop_last: bool = False,
    seed: int = 42,
    field_stats: dict[str, Any] | None = None,
    dataset_cls: type = BinaryTFRecordDataset,
) -> DataLoader:
    """Build a DataLoader for a dataset."""
    dataset = dataset_cls(
        files,
        field_entries,
        field_stats=field_stats,
        shuffle_files=shuffle_files,
        shuffle_buffer=shuffle_buffer,
        seed=seed,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=BatchCollator([e.field_name for e in field_entries]),
        drop_last=drop_last,
    )


def build_dataloaders(data_cfg: dict[str, Any], model_cfg: Any, train_cfg: Any, dataset_cls: type = BinaryTFRecordDataset) -> tuple[DataLoader, DataLoader | None, DataLoader | None]:
    """Build DataLoaders for training, validation, and test sets."""
    root: Path = Path(data_cfg["paths"]["tfrecord_root"])
    subs: dict[str, str] = data_cfg["split_subdirs"]
    train_files: list[Path] = collect_tfrecord_part_files(root / subs["train"] / "tfrecord")
    val_files: list[Path] = collect_tfrecord_part_files(root / subs["val"] / "tfrecord")
    test_files: list[Path] = collect_tfrecord_part_files(root / subs["test"] / "tfrecord")

    field_entries = list(model_cfg.embedding_fields.values())
    seed = train_cfg.seed
    loader_kw = dict(
        field_stats=model_cfg.field_stats,
        batch_size=train_cfg.data.batch_size,
        num_workers=train_cfg.data.num_workers,
        pin_memory=train_cfg.data.pin_memory,
        shuffle_buffer=train_cfg.data.shuffle_buffer,
        drop_last=train_cfg.data.drop_last,
    )
    train_loader = build_loader(train_files, field_entries, shuffle_files=data_cfg.get("shuffle_train_files", True), seed=seed, dataset_cls=dataset_cls, **loader_kw)
    validation_loader = build_loader(val_files, field_entries, shuffle_files=False, seed=seed + 101, dataset_cls=dataset_cls, **loader_kw) if val_files else None
    test_loader = build_loader(test_files, field_entries, shuffle_files=False, seed=seed + 202, dataset_cls=dataset_cls, **loader_kw) if test_files else None
    return train_loader, validation_loader, test_loader


def build_model_config(exp_cfg: dict[str, Any], config_class: type) -> Any:
    """Build a model configuration from a dictionary."""
    model_raw: dict[str, Any] = exp_cfg["model"]
    data_cfg: dict[str, Any] = exp_cfg["data"]
    enabled_field_entries, disabled_field_names = load_enabled_field_entries(model_raw)
    print(f"Disabled fields: {disabled_field_names}")

    model_cfg = config_class.from_dict(model_raw, enabled_field_entries)
    pos_map_json = Path(data_cfg["paths"]["nn_pos_map_json"])
    field_stats = load_field_stats(pos_map_json)
    target_size = load_target_size(pos_map_json)
    model_cfg.field_stats = field_stats
    model_cfg.target_size = target_size
    print(f"Target size: {target_size}")
    return model_cfg
