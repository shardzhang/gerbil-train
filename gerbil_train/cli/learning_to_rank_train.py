from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import torch
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from gerbil_train.data.learning_to_rank_dataset import build_ltr_dataloaders
from gerbil_train.losses.ranking import LOSS_CHOICES
from gerbil_train.models.learning_to_rank import DeepRankNet
from gerbil_train.trainer.learning_to_rank_trainer import LearningToRankTrainer

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).parent.parent.parent
print(f"Project root directory: {PROJECT_ROOT.resolve()}")
DEFAULT_DATASET_PATH = PROJECT_ROOT.parent / "data" / "MSLR-WEB10K.pt"
DEFAULT_CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints" / "ltr"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for learning-to-rank training."""
    parser = argparse.ArgumentParser(description="Train a learning-to-rank model")
    parser.add_argument(
        "--loss",
        choices=LOSS_CHOICES,
        default="lambdarank",
        help="Loss function to optimize",
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help="Path to the MSLR-WEB10K.pt file",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible comparison runs",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=DEFAULT_CHECKPOINT_DIR,
        help="Directory to save model checkpoints",
    )
    parser.add_argument(
        "--monitor",
        type=str,
        default="val_ndcg",
        help="Metric name used for early stopping and best-checkpoint selection",
    )
    parser.add_argument(
        "--monitor-mode",
        choices=("min", "max"),
        default="max",
        help="Whether smaller or larger monitored values are better",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=5,
        help="Early stopping patience measured in validation checks",
    )
    return parser.parse_args()


def main() -> None:
    """Run the learning-to-rank training entrypoint."""
    args = parse_args()
    loss_name = args.loss
    dataset_path = args.dataset_path
    checkpoint_dir = args.checkpoint_dir
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    model_path = checkpoint_dir / f"best_model_{loss_name}.pth"
    plot_path = checkpoint_dir / f"training_curves_{loss_name}.png"
    log_dir = checkpoint_dir / "log_dir" / loss_name

    print(f"Loading dataset from {dataset_path}")
    train_loader, val_loader, test_loader = build_ltr_dataloaders(dataset_path)
    print(
        f"Loaded {len(train_loader)} train / {len(val_loader)} val / {len(test_loader)} test queries"
    )
    print(f"Using loss: {loss_name}")
    print(f"Using seed: {args.seed}")

    model = DeepRankNet(input_dim=136)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)
    trainer = LearningToRankTrainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        gradient_clip_norm=None,
        monitor=args.monitor,
        monitor_mode=args.monitor_mode,
        patience=args.patience,
        best_checkpoint_path=model_path,
        best_metric=None,
        wait=0,
        seed=args.seed,
        log_dir=log_dir,
        plot_path=plot_path,
    )

    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        loss_name=loss_name,
        epochs=args.epochs,
    )
    print(f"Saved best model to {model_path.resolve()}")
    print(f"Saved training curves to {plot_path.resolve()}")

    trainer.load_checkpoint(str(model_path))
    test_metrics = trainer.evaluate(test_loader, ks=(1, 3, 5, 10))
    print(f"Test NDCG@1  = {test_metrics[1]:.4f}")
    print(f"Test NDCG@3  = {test_metrics[3]:.4f}")
    print(f"Test NDCG@5  = {test_metrics[5]:.4f}")
    print(f"Test NDCG@10 = {test_metrics[10]:.4f}")


if __name__ == "__main__":
    main()


# python -m gerbil_train.cli.learning_to_rank_train --epochs 1 --loss mse --seed 42 2>&1 | rg "Using loss|Using seed|^Epoch|Early stopping|Saved best model|Saved training curves|Test NDCG"
# python -m gerbil_train.cli.learning_to_rank_train --epochs 1 --loss ranknet --seed 42 2>&1 | rg "Using loss|Using seed|^Epoch|Early stopping|Saved best model|Saved training curves|Test NDCG"
# python -m gerbil_train.cli.learning_to_rank_train --epochs 1 --loss lambdarank --seed 42 2>&1 | rg "Using loss|Using seed|^Epoch|Early stopping|Saved best model|Saved training curves|Test NDCG"
# python -m gerbil_train.cli.learning_to_rank_train --epochs 1 --loss listmle --seed 42 2>&1 | rg "Using loss|Using seed|^Epoch|Early stopping|Saved best model|Saved training curves|Test NDCG"
# python -m gerbil_train.cli.learning_to_rank_train --epochs 1 --loss listnet --seed 42 2>&1 | rg "Using loss|Using seed|^Epoch|Early stopping|Saved best model|Saved training curves|Test NDCG"
