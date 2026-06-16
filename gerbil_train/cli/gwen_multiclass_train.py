"""Train GwEN (Group-wise Embedding Network) with TFRecord samples."""

from __future__ import annotations
from pathlib import Path
from typing import Any
import yaml
import torch
from torch.utils.data import DataLoader

from gerbil_train.config import (
    GwENFieldEntry,
    GwENModelConfig,
    GwENTrainConfig,
)
from gerbil_train.data.multi_tfrecord_dataset import MultiTFRecordDataset
from gerbil_train.data.tfrecord_dataset import (
    BatchCollator, FieldSpec, collect_tfrecord_part_files,
    count_tfrecord_records, load_field_specs,
    load_field_stats, load_target_size,
)
from gerbil_train.utils.config import load_experiment_config, parse_args
from gerbil_train.utils.run import create_run_dir, filter_enabled_fields, save_run_configs
from gerbil_train.models.gwen_multiclass_model import GwEN
from gerbil_train.trainer.gwen_multiclass_trainer import GwENTrainer

PROJECT_ROOT = Path(__file__).parent.parent.parent
CONFIG_PATH = PROJECT_ROOT / "configs/experiment/gwen_ml1m_multiclass.yaml"


def build_dataloaders(cfg: dict[str, Any], all_field_specs: list[FieldSpec]) -> tuple[DataLoader, DataLoader | None, DataLoader | None, int]:
    data_cfg = cfg["data"]
    train_cfg = cfg["train"]
    data_loader_cfg = train_cfg.get("data", {})
    model_cfg_raw = cfg["model"]

    field_enabled = {
        name: bool(entry.get("enabled", True))
        for name, entry in model_cfg_raw.get("embedding", {}).get("fields", {}).items()
    }
    enabled_specs = filter_enabled_fields(all_field_specs, field_enabled)
    field_names = [spec.name for spec in enabled_specs]

    root = Path(data_cfg["paths"]["tfrecord_root"])
    pos_map_json = Path(data_cfg["paths"]["nn_pos_map_json"])
    subs = data_cfg.get("split_subdirs", {"train": "train", "val": "val", "test": "test"})

    target_size = load_target_size(pos_map_json)
    field_stats = load_field_stats(pos_map_json)
    seed = int(train_cfg.get("seed", 42))

    train_files = collect_tfrecord_part_files(root / subs["train"] / "tfrecord")
    val_files = collect_tfrecord_part_files(root / subs["val"] / "tfrecord")
    test_files = collect_tfrecord_part_files(root / subs["test"] / "tfrecord")

    if not train_files:
        raise ValueError("No TFRecord files found for training split")

    collate_fn = BatchCollator(field_names)
    batch_size = int(data_loader_cfg.get("batch_size", 1024))
    num_workers = int(data_loader_cfg.get("num_workers", 0))
    pin_memory = bool(data_loader_cfg.get("pin_memory", False))
    drop_last = bool(data_loader_cfg.get("drop_last", False))
    prefetch_factor = int(data_loader_cfg.get("prefetch_factor", 2)) if num_workers > 0 else None

    loader_kwargs = dict(
        batch_size=batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=pin_memory, collate_fn=collate_fn, drop_last=drop_last,
    )
    if prefetch_factor is not None:
        loader_kwargs["prefetch_factor"] = prefetch_factor

    shuffle_buffer = int(data_loader_cfg.get("shuffle_buffer", 0))

    def make_loader(files, shuffle_files, extra_seed=0):
        return DataLoader(
            MultiTFRecordDataset(
                files, field_names, field_stats=field_stats,
                shuffle_files=shuffle_files, shuffle_buffer=shuffle_buffer,
                seed=seed + extra_seed,
            ),
            **loader_kwargs,
        )

    train_loader = make_loader(train_files, shuffle_files=bool(data_cfg.get("shuffle_train_files", True)))
    validation_loader = make_loader(val_files, shuffle_files=False, extra_seed=101) if val_files else None
    test_loader = make_loader(test_files, shuffle_files=False, extra_seed=202) if test_files else None
    return train_loader, validation_loader, test_loader, target_size


def build_model_config(raw_model_cfg: dict[str, Any], target_size: int, field_specs: list) -> GwENModelConfig:
    """Build the GwEN model config with auto-generated fields, then persist to YAML."""
    model_config_path = (PROJECT_ROOT / "configs/model/gwen_multiclass_model.yaml").resolve()
    raw_cfg = yaml.safe_load(model_config_path.read_text(encoding="utf-8"))
    default_emb_dim = int(raw_cfg.get("embedding", {}).get("default_emb_dim", 16))
    existing_fields = raw_cfg.get("embedding", {}).get("fields", {}) or {}

    all_entries: dict[str, GwENFieldEntry] = {}
    for spec in field_specs:
        existing = existing_fields.get(spec.name, {})
        emb_dim = int(existing.get("emb_dim", default_emb_dim))
        enabled = bool(existing.get("enabled", True))
        all_entries[spec.name] = GwENFieldEntry(
            f_index=spec.index,
            f_type=spec.field_type,
            vocab_size=int(spec.dim),
            emb_dim=emb_dim,
            enabled=enabled,
        )

    # Write YAML: all fields (including disabled)
    raw_cfg["target_size"] = int(target_size)
    raw_cfg["embedding"]["fields"] = {
        name: {"f_index": e.f_index, "f_type": e.f_type, "vocab_size": e.vocab_size, "emb_dim": e.emb_dim, "enabled": e.enabled}
        for name, e in sorted(all_entries.items(), key=lambda item: item[1].f_index)
    }
    with model_config_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(raw_cfg, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    print(f"Config written to {model_config_path}")

    # Return model config: only enabled fields
    enabled_entries = {name: e for name, e in all_entries.items() if e.enabled}
    model_cfg = GwENModelConfig.from_dict(raw_model_cfg, enabled_entries)
    model_cfg.target_size = int(target_size)
    return model_cfg


def main() -> None:
    args = parse_args(CONFIG_PATH)
    run_dir, checkpoint_path, plot_path = create_run_dir(PROJECT_ROOT / "checkpoints" / "gwen_ml1m_tfrecord")

    cfg = load_experiment_config(args.config)
    data_cfg = cfg["data"]
    model_cfg_raw = cfg["model"]
    train_cfg = GwENTrainConfig.from_dict(cfg["train"])
    train_cfg.checkpoint.path = str(checkpoint_path)
    train_cfg.logging.plot_path = str(plot_path)
    print(f"Training config | seed={train_cfg.seed} | epochs={train_cfg.epochs} | batch_size={train_cfg.data.batch_size}")
    print(f"Run dir: {run_dir}")
    print(f"Loading GwEN TFRecords from {data_cfg['paths']['tfrecord_root']}")

    all_field_specs = load_field_specs(data_cfg["paths"]["nn_pos_map_txt"])
    train_loader, validation_loader, test_loader, target_size = build_dataloaders(cfg, all_field_specs)

    model_cfg = build_model_config(model_cfg_raw, target_size, all_field_specs)
    model = GwEN(model_cfg)
    if train_cfg.compile.enabled:
        model = torch.compile(model, mode=train_cfg.compile.mode)
        print(f"Model compiled with torch.compile (mode={train_cfg.compile.mode})")
    trainer = GwENTrainer(model, train_cfg)
    trainer.set_profile_path(run_dir)
    root = Path(data_cfg["paths"]["tfrecord_root"])
    subs = data_cfg.get("split_subdirs", {"train": "train", "val": "val", "test": "test"})
    train_files = collect_tfrecord_part_files(root / subs["train"] / "tfrecord")
    trainer.set_total_train_samples(count_tfrecord_records(train_files), train_cfg.data.batch_size)
    trainer.fit(train_loader, validation_loader, test_loader)

    if test_loader is not None:
        test_metrics = trainer.evaluate(test_loader)
        print(f"Final test metrics: {test_metrics}")
    save_run_configs(args.config, run_dir, project_root=PROJECT_ROOT)


if __name__ == "__main__":
    main()

# python3 -m gerbil_train.cli.gwen_multiclass_train --config configs/experiment/gwen_ml1m_multiclass.yaml