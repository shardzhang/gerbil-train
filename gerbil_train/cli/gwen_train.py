"""Train GwEN (Group-wise Embedding Network) with TFRecord samples."""

from __future__ import annotations
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any
import yaml
from torch.utils.data import DataLoader

from gerbil_train.config import (
    GwENFieldEntry,
    GwENModelConfig,
    GwENTrainConfig,
)
from gerbil_train.data.gwen_tfrecord_dataset import (
    GwENBatchCollator,
    GwENTFRecordDataset,
    collect_tfrecord_part_files,
    load_gwen_field_specs,
    load_gwen_field_stats,
    load_target_size,
)
from gerbil_train.utils.config import load_experiment_config, parse_args
from gerbil_train.models.gwen import GwEN
from gerbil_train.trainer.gwen_trainer import GwENTrainer

PROJECT_ROOT = Path(__file__).parent.parent.parent
CONFIG_PATH = PROJECT_ROOT / "configs/experiment/gwen_ml1m_multiclass.yaml"


def filter_enabled_fields(field_specs: list, model_cfg_raw: dict[str, Any]) -> list:
    """Filter field specs based on the ``enabled`` flag in the model config."""
    field_enabled = {
        name: bool(entry.get("enabled", True))
        for name, entry in model_cfg_raw.get("embedding", {}).get("fields", {}).items()
    }
    disabled_fields = [name for name, enabled in field_enabled.items() if not enabled]
    if disabled_fields:
        print(f"Disabled fields ({len(disabled_fields)}): {disabled_fields}")
    return [spec for spec in field_specs if field_enabled.get(spec.name, True)]


def build_dataloaders(
    cfg: dict[str, Any],
    field_names: list[str],
    field_stats: dict[str, dict[int, tuple[float, float]]] | None = None,
) -> tuple[DataLoader, DataLoader | None, DataLoader | None, int]:
    data_cfg = cfg["data"]
    train_cfg = cfg["train"]
    data_loader_cfg = train_cfg.get("data", {})

    tfrecord_root = Path(data_cfg["paths"]["tfrecord_root"])
    nn_pos_map_json = Path(data_cfg["paths"]["nn_pos_map_json"])
    split_subdirs = data_cfg.get("split_subdirs", {"train": "train", "val": "val", "test": "test"})

    target_size = load_target_size(nn_pos_map_json)
    seed = int(train_cfg.get("seed", 42))

    train_files = collect_tfrecord_part_files(tfrecord_root / split_subdirs["train"] / "tfrecord")
    val_files = collect_tfrecord_part_files(tfrecord_root / split_subdirs["val"] / "tfrecord")
    test_files = collect_tfrecord_part_files(tfrecord_root / split_subdirs["test"] / "tfrecord")

    if not train_files:
        raise ValueError("No TFRecord files found for training split")

    collate_fn = GwENBatchCollator(field_names)
    batch_size = int(data_loader_cfg.get("batch_size", 1024))
    num_workers = int(data_loader_cfg.get("num_workers", 0))
    pin_memory = bool(data_loader_cfg.get("pin_memory", False))
    drop_last = bool(data_loader_cfg.get("drop_last", False))
    prefetch_factor = int(data_loader_cfg.get("prefetch_factor", 2)) if num_workers > 0 else None

    loader_kwargs = dict(
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
        drop_last=drop_last,
    )
    if prefetch_factor is not None:
        loader_kwargs["prefetch_factor"] = prefetch_factor

    train_loader = DataLoader(
        GwENTFRecordDataset(
            train_files,
            field_names,
            field_stats=field_stats,
            shuffle_files=bool(data_cfg.get("shuffle_train_files", True)),
            seed=seed,
        ),
        **loader_kwargs,
    )

    validation_loader = None
    if val_files and data_cfg.get("validation", {}).get("enabled", True):
        validation_loader = DataLoader(
            GwENTFRecordDataset(
                val_files,
                field_names,
                field_stats=field_stats,
                shuffle_files=False,
                seed=seed + 101,
            ),
            **loader_kwargs,
        )

    test_loader = None
    if test_files and data_cfg.get("test", {}).get("enabled", True):
        test_loader = DataLoader(
            GwENTFRecordDataset(
                test_files,
                field_names,
                field_stats=field_stats,
                shuffle_files=False,
                seed=seed + 202,
            ),
            **loader_kwargs,
        )
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

    run_id = datetime.now().strftime("%Y%m%d%H%M%S")
    run_dir = PROJECT_ROOT / "checkpoints" / "gwen_ml1m_tfrecord" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_experiment_config(args.config)
    data_cfg = cfg["data"]
    model_cfg_raw = cfg["model"]
    train_cfg = GwENTrainConfig.from_dict(cfg["train"])

    train_cfg.checkpoint.path = str(run_dir / "best_model.pth")
    train_cfg.logging.plot_path = str(run_dir / "training_curves.png")

    print("Training config | "
        f"seed={train_cfg.seed} | "
        f"epochs={train_cfg.epochs} | "
        f"batch_size={train_cfg.data.batch_size}"
    )
    print(f"Run dir: {run_dir}")
    print(f"Loading GwEN TFRecords from {data_cfg['paths']['tfrecord_root']}")

    all_field_specs = load_gwen_field_specs(data_cfg["paths"]["nn_pos_map_txt"])
    enabled_specs = filter_enabled_fields(all_field_specs, model_cfg_raw)

    field_names = [spec.name for spec in enabled_specs]
    field_stats = load_gwen_field_stats(data_cfg["paths"]["nn_pos_map_json"])
    train_loader, validation_loader, test_loader, target_size = build_dataloaders(cfg, field_names, field_stats)

    model_cfg = build_model_config(model_cfg_raw, target_size, all_field_specs)
    model = GwEN(model_cfg)

    if train_cfg.compile.enabled:
        import torch
        model = torch.compile(model, mode=train_cfg.compile.mode)
        print(f"Model compiled with torch.compile (mode={train_cfg.compile.mode})")

    trainer = GwENTrainer(model, train_cfg)
    trainer.fit(train_loader, validation_loader, test_loader)

    if test_loader is not None:
        test_metrics = trainer.evaluate(test_loader)
        print(f"Final test metrics: {test_metrics}")

    exp_cfg_path = Path(args.config)
    with open(exp_cfg_path, encoding="utf-8") as f:
        exp_raw = yaml.safe_load(f)
    shutil.copy2(str(exp_cfg_path), str(run_dir / "experiment.yaml"))
    for key in ("data", "model", "train"):
        sub_path = exp_raw.get(key)
        if sub_path:
            src = PROJECT_ROOT / sub_path
            if src.exists():
                shutil.copy2(str(src), str(run_dir / f"{key}.yaml"))
    print(f"Run artifacts saved to {run_dir}")


if __name__ == "__main__":
    main()

# python3 -m gerbil_train.cli.gwen_train --config configs/experiment/gwen_ml1m_multiclass.yaml