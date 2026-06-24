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
    BatchCollator, BinaryTFRecordDataset, collect_tfrecord_part_files,
    load_target_size, load_field_stats,
)
from gerbil_train.config.model_config import GwENModelConfig, FieldEntry, load_enabled_field_entries
from gerbil_train.config.train_config import GwENTrainConfig
from gerbil_train.models.gwen import GwENBinaryModel, GwENMulticlassModel
from gerbil_train.inference import Predictor

PROJECT_ROOT = Path(__file__).parent.parent.parent

MODEL_REGISTRY: dict[str, Any] = {}
MODEL_REGISTRY["gwen_binary"] = GwENBinaryModel
MODEL_REGISTRY["gwen_multiclass"] = GwENMulticlassModel


def build_loader(split_name: str, data_cfg: dict[str, Any], model_cfg: GwENModelConfig, train_cfg: GwENTrainConfig) -> DataLoader:
    root = Path(data_cfg["paths"]["tfrecord_root"])
    subs = data_cfg["split_subdirs"]
    files = collect_tfrecord_part_files(root / subs[split_name] / "tfrecord")
    field_entries = list(model_cfg.embedding_fields.values())
    kwargs = dict(
        field_stats=model_cfg.field_stats,
        batch_size=train_cfg.data.batch_size,
        num_workers=train_cfg.data.num_workers,
        pin_memory=train_cfg.data.pin_memory,
        shuffle_buffer=0,
        drop_last=False,
        seed=42,
    )
    dataset = BinaryTFRecordDataset(
        files, 
        field_entries,
        field_stats=kwargs["field_stats"],
        shuffle_files=False,
        shuffle_buffer=kwargs["shuffle_buffer"],
        seed=kwargs["seed"],
    )
    return DataLoader(
        dataset,
        batch_size=kwargs["batch_size"],
        shuffle=False,
        num_workers=kwargs["num_workers"],
        pin_memory=kwargs["pin_memory"],
        collate_fn=BatchCollator([e.field_name for e in field_entries]),
        drop_last=False,
    )


def build_model_config(exp_cfg: dict[str, Any]) -> GwENModelConfig:
    """Build a GwENModelConfig from the experiment configuration."""
    data_cfg = exp_cfg["data"]
    model_cfg_raw = exp_cfg["model"]
    enabled_entries, disabled = load_enabled_field_entries(model_cfg_raw)
    if disabled:
        print(f"Disabled fields: {disabled}")

    cfg = GwENModelConfig.from_dict(model_cfg_raw, enabled_entries)
    pos_map_json = Path(data_cfg["paths"]["nn_pos_map_json"])
    cfg.field_stats = load_field_stats(pos_map_json)
    cfg.target_size = load_target_size(pos_map_json)
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline inference")
    parser.add_argument("--config", type=Path, required=True, help="Experiment YAML path")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Model checkpoint path")
    parser.add_argument("--model-type", type=str, default="gwen_binary", choices=list(MODEL_REGISTRY.keys()), help="Model type")
    parser.add_argument("--split", type=str, default="test", choices=["val", "test"])
    parser.add_argument("--output", type=Path, default=None, help="Output predictions file")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    experiment_cfg = load_experiment_config(args.config)
    model_cfg = build_model_config(experiment_cfg)
    train_cfg = GwENTrainConfig.from_dict(experiment_cfg["train"])
    dataloader = build_loader(args.split, experiment_cfg["data"], model_cfg, train_cfg)

    model_cls = MODEL_REGISTRY[args.model_type]
    extra_kwargs = {}
    if args.model_type == "din":
        import yaml
        raw = yaml.safe_load(Path(args.config).read_text())
        bf = raw.get("model", {}).get("behavior_fields", [])
        extra_kwargs["behavior_fields"] = bf
    model = model_cls(model_cfg, **extra_kwargs)

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
    --checkpoint checkpoints/gwen_ml1m_binary/.../best_model.pth \
    --model-type gwen_binary \
    --split test \
    --output /tmp/predictions.tsv

Loaded checkpoint from checkpoints/.../best_model.pth
Wrote 800167 results to /tmp/predictions.tsv

Evaluation metrics (test set):
  auc: 0.7694
  ap: 0.5234
  gauc: 0.7632
  map: 0.5812
  mrr: 0.6123
"""