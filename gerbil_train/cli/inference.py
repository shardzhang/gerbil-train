"""Offline inference CLI for all model types.

Usage:
    python3 -m gerbil_train.cli.inference \\
        --config configs/2-gwen_ml1m_binary/experiment.yaml \\
        --checkpoint checkpoints/.../best_model.pth \\
        --split test \\
        --output /tmp/predictions.tsv
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from torch.utils.data import DataLoader

from gerbil_train.utils.config import load_experiment_config
from gerbil_train.data.tfrecord_dataset import (
    BatchCollator, BinaryTFRecordDataset, MultiTFRecordDataset,
    collect_tfrecord_part_files, load_target_size, load_field_stats,
)
from gerbil_train.config.model_config import ModelConfig, FieldEntry, load_enabled_field_entries
from gerbil_train.models.gwen import GwENBinaryModel, GwENMulticlassModel
from gerbil_train.inference import Predictor

PROJECT_ROOT = Path(__file__).parent.parent.parent

MODEL_REGISTRY: dict[str, dict[str, Any]] = {
    "gwen_binary": {"model": GwENBinaryModel, "config": ModelConfig, "dataset": BinaryTFRecordDataset},
    "gwen_multiclass": {"model": GwENMulticlassModel, "config": ModelConfig, "dataset": MultiTFRecordDataset},
}


def build_loader(
    split_name: str,
    data_cfg: dict[str, Any],
    field_entries: list[FieldEntry],
    field_stats: dict[str, Any] | None,
    batch_size: int,
    num_workers: int,
    pin_memory: bool = False,
    dataset_cls: type = BinaryTFRecordDataset,
) -> DataLoader:
    root = Path(data_cfg["paths"]["tfrecord_root"])
    subs = data_cfg["split_subdirs"]
    files = collect_tfrecord_part_files(root / subs[split_name] / "tfrecord")
    dataset = dataset_cls(
        files,
        field_entries,
        field_stats=field_stats,
        shuffle_files=False,
        shuffle_buffer=0,
        seed=42,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=BatchCollator([e.field_name for e in field_entries]),
        drop_last=False,
    )


def build_model_config(exp_cfg: dict[str, Any], config_class: type) -> Any:
    """Build a model config from the experiment configuration."""
    data_cfg = exp_cfg["data"]
    model_cfg_raw = exp_cfg["model"]
    enabled_entries, _ = load_enabled_field_entries(model_cfg_raw)
    model_cfg = config_class.from_dict(model_cfg_raw, enabled_entries)
    pos_map_json = Path(data_cfg["paths"]["nn_pos_map_json"])
    model_cfg.field_stats = load_field_stats(pos_map_json)
    model_cfg.target_size = load_target_size(pos_map_json)
    exp_cfg["_model_cfg"] = model_cfg
    return model_cfg


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline inference")
    parser.add_argument("--config", type=Path, required=True, help="Experiment YAML path")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Model checkpoint path")
    parser.add_argument("--model-type", type=str, default="gwen_binary", choices=list(MODEL_REGISTRY.keys()), help="Model type")
    parser.add_argument("--split", type=str, default="test", choices=["val", "test"])
    parser.add_argument("--output", type=Path, default=None, help="Output predictions file")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    model_info = MODEL_REGISTRY[args.model_type]
    exp_cfg = load_experiment_config(args.config)
    data_cfg = exp_cfg["data"]
    train_cfg = exp_cfg["train"]
    model_cfg = build_model_config(exp_cfg, model_info["config"])
    dataloader = build_loader(
        args.split,
        data_cfg,
        list(model_cfg.embedding_fields.values()),
        model_cfg.field_stats,
        train_cfg["data"]["batch_size"],
        train_cfg["data"]["num_workers"],
        train_cfg["data"].get("pin_memory", False),
        model_info["dataset"],
    )

    model = model_info["model"](model_cfg)
    predictor = Predictor(model, device=args.device)
    predictor.load_checkpoint(args.checkpoint)
    metrics = predictor.predict_and_eval(dataloader, output_path=args.output)
    print(f"\nEvaluation metrics ({args.split} set):")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()


"""
python3 -m gerbil_train.cli.inference \
--config configs/2-gwen_ml1m_binary/experiment.yaml \
--checkpoint checkpoints/gwen_ml1m_binary/20260624170859/best_model.pth \
--model-type gwen_binary \
--split test \
--output checkpoints/gwen_ml1m_binary/20260624170859/predictions.tsv
Field user_movie_rate (field_index=101)共享词表
Field user_movie_rate_15day (field_index=101)共享词表
Field user_movie_rate_1day (field_index=101)共享词表
Field user_movie_rate_3day (field_index=101)共享词表
Field user_movie_rate_7day (field_index=101)共享词表
Field user_genres_rate (field_index=103)共享词表
Field user_genres_rate_15day (field_index=103)共享词表
Field user_genres_rate_1day (field_index=103)共享词表
Field user_genres_rate_3day (field_index=103)共享词表
Field user_genres_rate_7day (field_index=103)共享词表
Loaded checkpoint from checkpoints/gwen_ml1m_binary/20260624170859/best_model.pth
Wrote 100022 results to checkpoints/gwen_ml1m_binary/20260624170859/predictions.tsv

Evaluation metrics (test set):
  auc: 0.7735
  ap: 0.7841
  gauc: 0.7735
  map: 0.7841
  mrr: 1.0000
"""