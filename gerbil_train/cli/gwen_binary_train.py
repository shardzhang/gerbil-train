"""Train GwEN binary classification model with TFRecord samples."""

from __future__ import annotations
from pathlib import Path
from typing import Any
import torch
from torch.utils.data import DataLoader

from gerbil_train.config import GwENModelConfig, GwENTrainConfig
from gerbil_train.data.binary_tfrecord_dataset import BinaryTFRecordDataset
from gerbil_train.data.tfrecord_dataset import (
    BatchCollator, FieldSpec, collect_tfrecord_part_files,
    count_tfrecord_records, load_field_specs, load_field_stats,
)
from gerbil_train.utils.config import load_experiment_config, parse_args
from gerbil_train.utils.run import build_field_entries, create_run_dir, filter_enabled_fields, save_run_configs
from gerbil_train.models.gwen import GwENBinary as GwENCTR
from gerbil_train.trainer.gwen_binary_trainer import GwENBinaryTrainer

PROJECT_ROOT = Path(__file__).parent.parent.parent
CONFIG_PATH = PROJECT_ROOT / "configs/experiment/gwen_ml1m_binary.yaml"


def build_dataloaders(cfg: dict[str, Any], all_specs: list[FieldSpec]) -> tuple[DataLoader, DataLoader | None, DataLoader | None]:
    data_cfg = cfg["data"]
    train_cfg = cfg["train"]
    dl_cfg = train_cfg.get("data", {})
    model_raw = cfg["model"]

    enabled = {
        name: bool(e.get("enabled", True))
        for name, e in model_raw.get("embedding", {}).get("fields", {}).items()
    }
    specs = filter_enabled_fields(all_specs, enabled)
    field_names = [s.name for s in specs]

    root = Path(data_cfg["paths"]["tfrecord_root"])
    pos_map_json = Path(data_cfg["paths"]["nn_pos_map_json"])
    subs = data_cfg.get("split_subdirs", {"train": "train", "val": "val", "test": "test"})
    field_stats = load_field_stats(pos_map_json)
    seed = int(train_cfg.get("seed", 42))

    train_files = collect_tfrecord_part_files(root / subs["train"] / "tfrecord")
    val_files = collect_tfrecord_part_files(root / subs["val"] / "tfrecord")
    test_files = collect_tfrecord_part_files(root / subs["test"] / "tfrecord")

    cf = BatchCollator(field_names)
    bs = int(dl_cfg.get("batch_size", 512))
    nw = int(dl_cfg.get("num_workers", 0))
    shuffle_buf = int(dl_cfg.get("shuffle_buffer", 0))
    kw = dict(batch_size=bs, shuffle=False, num_workers=nw, collate_fn=cf, drop_last=False)

    def dl(files, sf, es=0):
        return DataLoader(BinaryTFRecordDataset(
            files, field_names, field_stats=field_stats,
            shuffle_files=sf, shuffle_buffer=shuffle_buf, seed=seed + es,
        ), **kw)

    return dl(train_files, True), dl(val_files, False, 101) if val_files else None, dl(test_files, False, 202) if test_files else None


def build_model_config(raw: dict[str, Any], field_specs: list[FieldSpec]) -> GwENModelConfig:
    cfg_path = (PROJECT_ROOT / "configs/model/gwen_binary_model.yaml").resolve()
    entries, _ = build_field_entries(cfg_path, field_specs)
    return GwENModelConfig.from_dict(raw, entries)


def main() -> None:
    args = parse_args(CONFIG_PATH)
    run_dir, ckpt_path, plot_path = create_run_dir(PROJECT_ROOT / "checkpoints" / "gwen_binary_ml1m")

    cfg = load_experiment_config(args.config)
    model_raw = cfg["model"]
    train_cfg = GwENTrainConfig.from_dict(cfg["train"])
    train_cfg.checkpoint.path = str(ckpt_path)
    train_cfg.logging.plot_path = str(plot_path)
    print(f"Training config | seed={train_cfg.seed} | epochs={train_cfg.epochs} | batch_size={train_cfg.data.batch_size}")
    print(f"Training config | train_path={cfg['data']['paths']['tfrecord_root']}")
    print(f"Run dir: {run_dir}")

    all_specs = load_field_specs(cfg["data"]["paths"]["nn_pos_map_txt"])
    train_loader, val_loader, test_loader = build_dataloaders(cfg, all_specs)

    model_cfg = build_model_config(model_raw, all_specs)
    model = GwENCTR(model_cfg)
    if train_cfg.compile.enabled:
        model = torch.compile(model, mode=train_cfg.compile.mode)
        print(f"Model compiled with torch.compile (mode={train_cfg.compile.mode})")
    trainer = GwENBinaryTrainer(model, train_cfg)
    trainer.set_profile_path(run_dir)
    root = Path(cfg["data"]["paths"]["tfrecord_root"])
    train_files = collect_tfrecord_part_files(root / cfg["data"].get("split_subdirs", {}).get("train", "train") / "tfrecord")
    trainer.set_total_train_samples(count_tfrecord_records(train_files), train_cfg.data.batch_size)
    trainer.fit(train_loader, val_loader, test_loader)

    if test_loader is not None:
        print(f"Final test metrics: {trainer.evaluate(test_loader)}")
    save_run_configs(args.config, run_dir, project_root=PROJECT_ROOT)


if __name__ == "__main__":
    main()
