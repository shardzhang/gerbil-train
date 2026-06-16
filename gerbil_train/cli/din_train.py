"""Train DIN (Deep Interest Network) model with TFRecord samples."""

from __future__ import annotations
from pathlib import Path
from typing import Any
import torch
from torch.utils.data import DataLoader

from gerbil_train.config import GwENFieldEntry, GwENModelConfig, GwENTrainConfig
from gerbil_train.data.binary_tfrecord_dataset import BinaryTFRecordDataset
from gerbil_train.data.tfrecord_dataset import (
    BatchCollator, collect_tfrecord_part_files,
    count_tfrecord_records, load_field_specs,
)
from gerbil_train.utils.config import load_experiment_config, parse_args
from gerbil_train.utils.run import create_run_dir, filter_enabled_fields, save_run_configs
from gerbil_train.models.din import DIN
from gerbil_train.trainer.din_trainer import DINTrainer
import yaml

PROJECT_ROOT = Path(__file__).parent.parent.parent
CONFIG_PATH = PROJECT_ROOT / "configs/experiment/din_ml1m.yaml"


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
    return dl(train_files, shuffle_files=True), dl(val_files, False, 101) if val_files else None, dl(test_files, False, 202) if test_files else None


def build_model_config(raw: dict[str, Any], field_specs: list) -> tuple[GwENModelConfig, str]:
    cfg_path = (PROJECT_ROOT / "configs/model/din.yaml").resolve()
    raw_cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    default_emb = int(raw_cfg.get("embedding", {}).get("default_emb_dim", 16))
    existing = raw_cfg.get("embedding", {}).get("fields", {}) or {}
    behavior_field = str(raw_cfg.get("behavior_field", ""))

    entries: dict[str, GwENFieldEntry] = {}
    for spec in field_specs:
        ex = existing.get(spec.name, {})
        entries[spec.name] = GwENFieldEntry(
            f_index=spec.index, f_type=spec.field_type, vocab_size=int(spec.dim),
            emb_dim=int(ex.get("emb_dim", default_emb)),
            enabled=bool(ex.get("enabled", True)),
        )

    raw_cfg["embedding"]["fields"] = {
        n: {"f_index": e.f_index, "f_type": e.f_type, "vocab_size": e.vocab_size, "emb_dim": e.emb_dim, "enabled": e.enabled}
        for n, e in sorted(entries.items(), key=lambda x: x[1].f_index)
    }
    with open(cfg_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(raw_cfg, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    print(f"Config written to {cfg_path}")

    enabled_entries = {n: e for n, e in entries.items() if e.enabled}
    return GwENModelConfig.from_dict(raw, enabled_entries), behavior_field


def main() -> None:
    args = parse_args(CONFIG_PATH)
    run_dir, ckpt_path, plot_path = create_run_dir(PROJECT_ROOT / "checkpoints" / "din_ml1m")

    cfg = load_experiment_config(args.config)
    model_raw = cfg["model"]
    train_cfg = GwENTrainConfig.from_dict(cfg["train"])
    train_cfg.checkpoint.path = str(ckpt_path)
    train_cfg.logging.plot_path = str(plot_path)

    print(f"Training config | seed={train_cfg.seed} | epochs={train_cfg.epochs} | batch_size={train_cfg.data.batch_size}")
    print(f"Run dir: {run_dir}")

    all_specs = load_field_specs(cfg["data"]["paths"]["nn_pos_map_txt"])
    field_names = [s.name for s in all_specs]
    train_loader, val_loader, test_loader = build_dataloaders(cfg, field_names)

    model_cfg, behavior_field = build_model_config(model_raw, all_specs)
    model = DIN(model_cfg, behavior_field)
    if train_cfg.compile.enabled:
        model = torch.compile(model, mode=train_cfg.compile.mode)
        print(f"Model compiled with torch.compile (mode={train_cfg.compile.mode})")
    trainer = DINTrainer(model, train_cfg)
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
