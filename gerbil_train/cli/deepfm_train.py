"""Train a DeepFM model on TFRecord samples."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from gerbil_train.config.train_config import DeepFMConfig, DeepFMTrainConfig
from gerbil_train.data.tfrecord_binary_dataset import BinaryTFRecordDataset
from gerbil_train.data.tfrecord_dataset import (
    BatchCollator, collect_tfrecord_part_files,
    load_field_specs,
)
from gerbil_train.models.deepfm import DeepFM
from gerbil_train.trainer.deepfm_trainer import DeepFMTrainer
from gerbil_train.utils.config import load_experiment_config, parse_args
from gerbil_train.utils.run import build_field_entries, create_run_dir, filter_enabled_fields, save_run_configs

PROJECT_ROOT = Path(__file__).parent.parent.parent
CONFIG_PATH = PROJECT_ROOT / "configs/experiment/deepfm_ml1m.yaml"


def build_dataloaders(
    cfg: dict[str, Any],
    field_names: list[str],
) -> tuple[DataLoader, DataLoader | None, DataLoader | None]:
    data_cfg = cfg["data"]
    train_cfg = cfg["train"]
    dl_cfg = train_cfg.get("data", {})

    root = Path(data_cfg["paths"]["tfrecord_root"])
    subs = data_cfg.get("split_subdirs", {"train": "train", "val": "val", "test": "test"})

    train_files = collect_tfrecord_part_files(root / subs["train"] / "tfrecord")
    val_files = collect_tfrecord_part_files(root / subs["val"] / "tfrecord")
    test_files = collect_tfrecord_part_files(root / subs["test"] / "tfrecord")

    cf = BatchCollator(field_names)
    bs = int(dl_cfg.get("batch_size", 512))
    nw = int(dl_cfg.get("num_workers", 0))
    shuffle_buf = int(dl_cfg.get("shuffle_buffer", 0))
    kw = dict(batch_size=bs, shuffle=False, num_workers=nw, collate_fn=cf, drop_last=False)

    def dl(files, sf, es=0):
        return DataLoader(
            BinaryTFRecordDataset(files, field_names, shuffle_files=sf, shuffle_buffer=shuffle_buf, seed=seed + es),
            **kw,
        )

    seed = int(train_cfg.get("seed", 42))
    return dl(train_files, True), dl(val_files, False, 101) if val_files else None, dl(test_files, False, 202) if test_files else None


def build_model_config(raw: dict[str, Any], all_specs: list) -> DeepFMConfig:
    cfg_path = (PROJECT_ROOT / "configs/model/deepfm.yaml").resolve()
    entries, _ = build_field_entries(cfg_path, all_specs)
    return DeepFMConfig.from_dict({**raw, "embedding_fields": entries})


def main() -> None:
    args = parse_args(CONFIG_PATH)
    run_dir, ckpt_path, plot_path = create_run_dir(PROJECT_ROOT / "checkpoints" / "deepfm_ml1m")

    cfg = load_experiment_config(args.config)
    model_raw = cfg["model"]
    train_cfg = DeepFMTrainConfig.from_dict(cfg["train"])
    train_cfg.checkpoint.path = str(ckpt_path)
    train_cfg.logging.plot_path = str(plot_path)
    print(f"Training config | seed={train_cfg.seed} | epochs={train_cfg.epochs} | batch_size={train_cfg.trainer.batch_size}")
    print(f"Run dir: {run_dir}")

    all_specs = load_field_specs(cfg["data"]["paths"]["nn_pos_map_txt"])
    field_names = [s.f_name for s in all_specs]
    train_loader, val_loader, test_loader = build_dataloaders(cfg, field_names)

    model_cfg = build_model_config(model_raw, all_specs)
    model = DeepFM(model_cfg)
    if train_cfg.compile.enabled:
        model = torch.compile(model, mode=train_cfg.compile.mode)
        print(f"Model compiled with torch.compile (mode={train_cfg.compile.mode})")
    trainer = DeepFMTrainer(model, train_cfg)
    trainer.setup_total_train_samples(cfg["data"], train_cfg.trainer.batch_size)
    trainer.fit(train_loader, val_loader, test_loader)

    if test_loader is not None:
        print(f"Final test metrics: {trainer.evaluate(test_loader)}")
    save_run_configs(args.config, run_dir, project_root=PROJECT_ROOT)


if __name__ == "__main__":
    main()
