"""Train GwEN (Group-wise Embedding Network) with TFRecord samples."""

from __future__ import annotations
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from gerbil_train.utils.config import load_experiment_config, parse_args
from gerbil_train.utils.run import create_run_dir, save_run_configs
from gerbil_train.data.tfrecord_dataset import BatchCollator, MultiTFRecordDataset, load_target_size, load_field_stats, FieldEntry, collect_tfrecord_part_files
from gerbil_train.config.model_config import GwENModelConfig, load_enabled_field_entries
from gerbil_train.config.train_config import GwENTrainConfig
from gerbil_train.models.gwen import GwENMulticlassModel
from gerbil_train.trainer.gwen_multiclass_trainer import GwENMultiTrainer

PROJECT_ROOT = Path(__file__).parent.parent.parent
CONFIG_PATH: Path = PROJECT_ROOT / "configs/1-gwen_ml1m_multiclass/experiment.yaml"


def _build_loader(
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
) -> DataLoader:
    """Build dataloader for a given set of files and field entries."""
    dataset = MultiTFRecordDataset(
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


def build_dataloaders(data_cfg: dict[str, Any], model_cfg: GwENModelConfig, train_cfg: GwENTrainConfig) -> tuple[DataLoader, DataLoader | None, DataLoader | None]:
    """Build dataloaders for training, validation, and test sets."""
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
        drop_last=train_cfg.data.drop_last
    )
    train_loader: DataLoader = _build_loader(train_files, field_entries, shuffle_files=data_cfg.get("shuffle_train_files", True), seed=seed, **loader_kw)
    validation_loader: DataLoader | None = _build_loader(val_files, field_entries, shuffle_files=False, seed=seed + 101, **loader_kw) if val_files else None
    test_loader: DataLoader | None = _build_loader(test_files, field_entries, shuffle_files=False, seed=seed + 202, **loader_kw) if test_files else None  
    return train_loader, validation_loader, test_loader


def build_model_config(model_cfg: dict[str, Any], data_cfg: dict[str, Any]) -> GwENModelConfig:
    """Build model config with enabled fields and field stats."""
    enabled_field_entries, disabled_field_names = load_enabled_field_entries(model_cfg)
    print(f"Disabled fields: {disabled_field_names}")
    
    model_cfg = GwENModelConfig.from_dict(model_cfg, enabled_field_entries)
    pos_map_json = Path(data_cfg["paths"]["nn_pos_map_json"])
    field_stats = load_field_stats(pos_map_json)
    target_size = load_target_size(pos_map_json)
    model_cfg.field_stats = field_stats
    model_cfg.target_size = target_size
    print(f"Target size: {target_size}")
    return model_cfg


def main() -> None:
    args = parse_args(CONFIG_PATH)
    exp_cfg: dict[str, Any] = load_experiment_config(args.config)
    data_cfg: dict[str, Any] = exp_cfg["data"]
    model_cfg: GwENModelConfig = build_model_config(exp_cfg["model"], data_cfg)
    
    run_dir, checkpoint_path, plot_path = create_run_dir(PROJECT_ROOT / "checkpoints" / "gwen_ml1m_multiclass")
    train_cfg: GwENTrainConfig = GwENTrainConfig.from_dict(exp_cfg["train"])
    train_cfg.checkpoint.path = str(checkpoint_path)
    train_cfg.logging.plot_path = str(plot_path)
    print(f"Training config | seed={train_cfg.seed} | epochs={train_cfg.epochs} | batch_size={train_cfg.data.batch_size}")
    print(f"Run dir: {run_dir}")
    print(f"Loading GwEN TFRecords from {data_cfg['paths']['tfrecord_root']}")

    train_loader, validation_loader, test_loader = build_dataloaders(data_cfg, model_cfg, train_cfg)
    model = GwENMulticlassModel(model_cfg)
    if train_cfg.compile.enabled:
        model = torch.compile(model, mode=train_cfg.compile.mode)
        print(f"Model compiled with torch.compile (mode={train_cfg.compile.mode})")
    trainer = GwENMultiTrainer(model, train_cfg, data_cfg)
    trainer.fit(train_loader, validation_loader, test_loader)

    if test_loader is not None:
        test_metrics = trainer.evaluate(test_loader)
        print(f"Final test metrics: {test_metrics}")
    save_run_configs(args.config, run_dir, project_root=PROJECT_ROOT)


if __name__ == "__main__":
    main()

# python3 -m gerbil_train.cli.gwen_multiclass_train --config configs/1-gwen_ml1m_multiclass/experiment.yaml
