"""Run the learning-to-rank training."""

from __future__ import annotations

from pathlib import Path

from gerbil_train.data.learning_to_rank_dataset import build_ltr_dataloaders
from gerbil_train.models.learning_to_rank import DeepRankNet
from gerbil_train.trainer.learning_to_rank_trainer import LearningToRankTrainer
from gerbil_train.utils.config import load_experiment_config, parse_args

PROJECT_ROOT = Path(__file__).parent.parent.parent
CONFIG_PATH = PROJECT_ROOT / "configs/experiment/learning_to_rank.yaml"


def main() -> None:
    args = parse_args(CONFIG_PATH)

    # 0. load config and print basic info
    cfg = load_experiment_config(args.config)
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]["model"]
    train_cfg = cfg["train"]
    print(f"Using seed: {train_cfg['seed']}")
    print(f"Using epochs: {train_cfg['epochs']}")
    print(f"Using loss: {str(train_cfg['loss_name'])}")

    # 1. data loaders
    dataset_path = Path(data_cfg["paths"]["dataset"])
    print(f"Loading dataset from {dataset_path}")
    train_loader, val_loader, test_loader = build_ltr_dataloaders(dataset_path)
    print(f"Loaded {len(train_loader)} train / {len(val_loader)} val / {len(test_loader)} test queries")

    # 2. model, trainer, and training
    model = DeepRankNet(model_cfg)
    trainer = LearningToRankTrainer(model, train_cfg)
    trainer.fit(train_loader, val_loader)

    # 3. evaluation
    trainer.load_checkpoint(str(trainer.best_checkpoint_path))
    test_metrics = trainer.evaluate(test_loader, ks=(1, 3, 5, 10))
    print("Final test metrics: ", test_metrics)

if __name__ == "__main__":
    main()

# python -m gerbil_train.cli.learning_to_rank_train --config configs/experiment/learning_to_rank.yaml
