"""Run the learning-to-rank training."""

from __future__ import annotations

from pathlib import Path

import torch
from gerbil_train.config.train_config import DeepFMTrainConfig, LTRConfig
from gerbil_train.data.learning_to_rank_dataset import build_ltr_dataloaders
from gerbil_train.models.learning_to_rank import DeepRankNet
from gerbil_train.trainer.learning_to_rank_trainer import LearningToRankTrainer
from gerbil_train.utils.config import load_experiment_config, parse_args

PROJECT_ROOT = Path(__file__).parent.parent.parent
CONFIG_PATH = PROJECT_ROOT / "configs/experiment/learning_to_rank.yaml"


def main() -> None:
    args = parse_args(CONFIG_PATH)
    cfg = load_experiment_config(args.config)
    data_cfg = cfg["data"]
    model_cfg = LTRConfig.from_dict(cfg["model"])
    train_cfg = DeepFMTrainConfig.from_dict(cfg["train"])
    print(f"Using seed: {train_cfg.seed}")
    print(f"Using epochs: {train_cfg.epochs}")

    dataset_path = Path(data_cfg["paths"]["dataset"])
    print(f"Loading dataset from {dataset_path}")
    train_loader, val_loader, test_loader = build_ltr_dataloaders(dataset_path)
    print(f"Loaded {len(train_loader)} train / {len(val_loader)} val / {len(test_loader)} test queries")

    model = DeepRankNet(model_cfg)
    if train_cfg.compile.enabled:
        model = torch.compile(model, mode=train_cfg.compile.mode)
        print(f"Model compiled with torch.compile (mode={train_cfg.compile.mode})")
    trainer = LearningToRankTrainer(model, train_cfg)
    trainer.fit(train_loader, val_loader)

    trainer.load_checkpoint(str(trainer.best_checkpoint_path))
    test_metrics = trainer.evaluate(test_loader, ks=(1, 3, 5, 10))
    print("Final test metrics: ", test_metrics)


if __name__ == "__main__":
    main()
