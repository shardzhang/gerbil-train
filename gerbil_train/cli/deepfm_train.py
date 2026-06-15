"""Train a DeepFM model on MovieLens 1M."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from torch.utils.data import DataLoader

from gerbil_train.data.deepfm_dataset import DeepFMDataset
from gerbil_train.data.deepfm_dataset import DeepFMRankingDataset
from gerbil_train.models.deepfm import DeepFM
from gerbil_train.trainer.deepfm_trainer import DeepFMTrainer
from gerbil_train.utils.config import load_experiment_config, parse_args

PROJECT_ROOT = Path(__file__).parent.parent.parent
CONFIG_PATH = PROJECT_ROOT / "configs/experiment/deepfm_ml1m.yaml"


def collate_single_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Return the single ranking group emitted by batch_size=1 loaders."""
    return batch[0]


def build_dataloaders(
    cfg: dict[str, Any],
) -> tuple[DataLoader, DataLoader | None, DataLoader | None]:
    """Build DeepFM train, validation, and test dataloaders."""
    data_cfg = cfg["data"]
    trainer_cfg = cfg["train"]["trainer"]
    model_cfg = cfg["model"]["model"]
    train_batch_size = int(trainer_cfg.get("batch_size", 256))
    num_workers = int(trainer_cfg.get("num_workers", 0))
    pin_memory = bool(trainer_cfg.get("pin_memory", False))
    drop_last = bool(trainer_cfg.get("drop_last", False))

    ratings_path = data_cfg["paths"]["interactions"]
    train_dataset = DeepFMDataset(ratings_path, split="train")
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )

    validation_loader = None
    test_loader = None
    validation_cfg = data_cfg.get("validation", {})
    if validation_cfg.get("enabled", False):
        num_negatives = int(validation_cfg.get("num_negatives", 99))
        validation_loader = DataLoader(
            DeepFMRankingDataset(
                ratings_path, split="validation", num_negatives=num_negatives
            ),
            batch_size=1,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
            collate_fn=collate_single_batch,
        )
        test_loader = DataLoader(
            DeepFMRankingDataset(
                ratings_path, split="test", num_negatives=num_negatives
            ),
            batch_size=1,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
            collate_fn=collate_single_batch,
        )

    return train_loader, validation_loader, test_loader


def main() -> None:
    args = parse_args(CONFIG_PATH)
    cfg = load_experiment_config(args.config)
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]["model"]
    train_cfg = cfg["train"]

    print(
        "Training config | "
        f"seed={train_cfg['seed']} | "
        f"epochs={train_cfg['epochs']} | "
        f"batch_size={train_cfg['trainer']['batch_size']}"
    )
    print(f"Loading DeepFM dataset from {data_cfg['paths']['interactions']}")

    train_loader, validation_loader, test_loader = build_dataloaders(cfg)
    model = DeepFM(model_cfg)
    trainer = DeepFMTrainer(model, train_cfg)
    trainer.fit(train_loader, validation_loader, test_loader)

    if test_loader is not None:
        test_metrics = trainer.evaluate(test_loader)
        print(f"Final test metrics: {test_metrics}")


if __name__ == "__main__":
    main()
