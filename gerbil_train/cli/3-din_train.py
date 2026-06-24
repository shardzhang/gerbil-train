"""Train DIN (Deep Interest Network) model with TFRecord samples."""

from __future__ import annotations
from pathlib import Path
from typing import Any
import torch
from torch.utils.data import DataLoader

from gerbil_train.config.model_config import DINModelConfig, load_enabled_field_entries
from gerbil_train.config.train_config import GwENTrainConfig
from gerbil_train.data.tfrecord_dataset import BinaryTFRecordDataset
from gerbil_train.data.tfrecord_dataset import (
    BatchCollator, collect_tfrecord_part_files,
    load_field_specs, load_target_size, load_field_stats,
)
from gerbil_train.utils.config import load_experiment_config, parse_args
from gerbil_train.utils.run import create_run_dir, save_run_configs
from gerbil_train.models.din import DIN
from gerbil_train.trainer.din_trainer import DINTrainer

PROJECT_ROOT = Path(__file__).parent.parent.parent
CONFIG_PATH = PROJECT_ROOT / "configs/3-din/din_ml1m_exp.yaml"


def build_dataloaders(cfg: dict[str, Any], field_names: list[str]) -> tuple[DataLoader, DataLoader | None, DataLoader | None]:
    data_cfg = cfg["data"]
    train_cfg = cfg["train"]
    dl_cfg = train_cfg.get("data", {})

    root = Path(data_cfg["paths"]["tfrecord_root"])
    subs = data_cfg.get("split_subdirs", {"train": "train", "val": "val", "test": "test"})
    field_stats = None

    train_files = collect_tfrecord_part_files(root / subs["train"] / "tfrecord")
    val_files = collect_tfrecord_part_files(root / subs["val"] / "tfrecord")
    test_files = collect_tfrecord_part_files(root / subs["test"] / "tfrecord")

    cf = BatchCollator(field_names)
    bs = int(dl_cfg.get("batch_size", 512))
    nw = int(dl_cfg.get("num_workers", 0))
    shuffle_buf = int(dl_cfg.get("shuffle_buffer", 0))
    kw = dict(batch_size=bs, shuffle=False, num_workers=nw, collate_fn=cf, drop_last=False)

    def dl(files, sf, es=0):
        return DataLoader(BinaryTFRecordDataset(files, field_names, field_stats=field_stats, shuffle_files=sf, shuffle_buffer=shuffle_buf, seed=seed + es), **kw)

    seed = int(train_cfg.get("seed", 42))
    return dl(train_files, True), dl(val_files, False, 101) if val_files else None, dl(test_files, False, 202) if test_files else None


def build_model_config(exp_cfg: dict[str, Any]) -> DINModelConfig:
    model_raw: dict[str, Any] = exp_cfg["model"]
    data_cfg: dict[str, Any] = exp_cfg["data"]
    entries, disabled_field_names = load_enabled_field_entries(model_raw)
    print(f"Disabled fields: {disabled_field_names}")

    cfg = DINModelConfig.from_dict(model_raw, entries)
    pos_map_json = Path(data_cfg["paths"]["nn_pos_map_json"])
    cfg.field_stats = load_field_stats(pos_map_json)
    cfg.target_size = load_target_size(pos_map_json)
    print(f"Target size: {cfg.target_size}")
    return cfg


def main() -> None:
    args = parse_args(CONFIG_PATH)
    run_dir, ckpt_path, plot_path = create_run_dir(PROJECT_ROOT / "checkpoints" / "din_ml1m")

    cfg = load_experiment_config(args.config)
    train_cfg = GwENTrainConfig.from_dict(cfg["train"])
    train_cfg.checkpoint.path = str(ckpt_path)
    train_cfg.logging.plot_path = str(plot_path)

    print(f"Training config | seed={train_cfg.seed} | epochs={train_cfg.epochs} | batch_size={train_cfg.data.batch_size}")
    print(f"Run dir: {run_dir}")

    all_specs = load_field_specs(cfg["data"]["paths"]["nn_pos_map_txt"])
    field_names = [s.field_name for s in all_specs]
    train_loader, val_loader, test_loader = build_dataloaders(cfg, field_names)

    model_cfg = build_model_config(cfg)
    model = DIN(model_cfg)
    if train_cfg.compile.enabled:
        model = torch.compile(model, mode=train_cfg.compile.mode)
        print(f"Model compiled with torch.compile (mode={train_cfg.compile.mode})")
    trainer = DINTrainer(model, train_cfg, cfg["data"])
    trainer.fit(train_loader, val_loader, test_loader)

    if test_loader is not None:
        print(f"Final test metrics: {trainer.evaluate(test_loader)}")
    save_run_configs(args.config, run_dir, project_root=PROJECT_ROOT)


if __name__ == "__main__":
    main()
