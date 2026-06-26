"""Run the shared-bottom two-tower training."""

from __future__ import annotations

from typing import Any
from pathlib import Path
import torch
from torch.utils.data import DataLoader

from gerbil_train.config.train_config import SBTTConfig
from gerbil_train.data.shared_bottom_two_tower_dataset import (
    ExplicitDataset,
    ImplicitDataset,
    RankingTestDataset,
    RankingValidationDataset,
)
from gerbil_train.models.shared_bottom_two_tower import SharedBottomTwoTower
from gerbil_train.trainer.shared_bottom_two_tower_trainer import (
    SharedBottomTwoTowerTrainer,
)
from gerbil_train.utils.config import load_experiment_config, parse_args

PROJECT_ROOT = Path(__file__).parent.parent.parent
CONFIG_PATH = PROJECT_ROOT / "configs/experiment/shared_bottom_two_tower.yaml"


def collate_single_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Return the single validation item emitted by batch_size=1 loaders."""
    return batch[0]


def build_dataloaders(
    cfg: dict[str, Any],
) -> tuple[DataLoader | None, DataLoader | None, DataLoader | None, DataLoader | None]:
    """Build dataloaders for the shared-bottom two-tower training pipeline.

    :param cfg: Fully resolved experiment configuration dictionary
    :return: Tuple of ``(implicit_loader, explicit_loader, validation_loader, test_loader)``
    """
    data_cfg = cfg["data"]
    loader_cfg = data_cfg.get("loader")
    if loader_cfg is None:
        trainer_cfg = cfg["train"]["trainer"]
        loader_cfg = {
            "train_batch_size": trainer_cfg.get("batch_size", 256),
            "num_workers": trainer_cfg.get("num_workers", 0),
            "pin_memory": trainer_cfg.get("pin_memory", False),
            "drop_last": trainer_cfg.get("drop_last", False),
        }

    implicit_loader = None
    explicit_loader = None
    validation_loader = None
    test_loader = None

    if data_cfg.get("sampling", {}).get("implicit", {}).get("negative_sampling", False):
        implicit_dataset = ImplicitDataset(
            data_path=data_cfg["paths"]["train_implicit"],
            query_input_dim=cfg["model"]["model"]["query_input_dim"],
            item_input_dim=cfg["model"]["model"]["item_input_dim"],
            num_negatives=data_cfg["sampling"]["implicit"]["num_negatives"],
            split="train",
        )
        implicit_loader = DataLoader(
            implicit_dataset,
            batch_size=loader_cfg["train_batch_size"],
            shuffle=True,
            num_workers=loader_cfg["num_workers"],
            pin_memory=loader_cfg["pin_memory"],
            drop_last=loader_cfg["drop_last"],
        )

    explicit_dataset = ExplicitDataset(
        data_path=data_cfg["paths"]["train_explicit"],
        query_input_dim=cfg["model"]["model"]["query_input_dim"],
        item_input_dim=cfg["model"]["model"]["item_input_dim"],
        split="train",
    )
    explicit_loader = DataLoader(
        explicit_dataset,
        batch_size=loader_cfg["train_batch_size"],
        shuffle=True,
        num_workers=loader_cfg["num_workers"],
        pin_memory=loader_cfg["pin_memory"],
        drop_last=loader_cfg["drop_last"],
    )

    validation_cfg = data_cfg.get("validation", {})
    if validation_cfg.get("enabled", False):
        validation_dataset = RankingValidationDataset(
            data_path=data_cfg["paths"]["train_explicit"],
            query_input_dim=cfg["model"]["model"]["query_input_dim"],
            item_input_dim=cfg["model"]["model"]["item_input_dim"],
            num_negatives=int(validation_cfg.get("num_negatives", 99)),
        )
        validation_loader = DataLoader(
            validation_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=loader_cfg["num_workers"],
            pin_memory=loader_cfg["pin_memory"],
            drop_last=False,
            collate_fn=collate_single_batch,
        )

        test_dataset = RankingTestDataset(
            data_path=data_cfg["paths"]["train_explicit"],
            query_input_dim=cfg["model"]["model"]["query_input_dim"],
            item_input_dim=cfg["model"]["model"]["item_input_dim"],
            num_negatives=int(validation_cfg.get("num_negatives", 99)),
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=loader_cfg["num_workers"],
            pin_memory=loader_cfg["pin_memory"],
            drop_last=False,
            collate_fn=collate_single_batch,
        )

    return implicit_loader, explicit_loader, validation_loader, test_loader


def main() -> None:
    args = parse_args(CONFIG_PATH)

    # 0. load config and print basic info
    cfg = load_experiment_config(args.config)
    data_cfg = cfg["data"]
    model_cfg = SBTTConfig.from_dict(cfg["model"]["model"])
    train_cfg = cfg["train"]
    print(
        "Training config | "
        f"seed={train_cfg['seed']} | "
        f"implicit_epochs={train_cfg['trainer']['implicit_epochs']} | "
        f"explicit_epochs={train_cfg['trainer']['explicit_epochs']}"
    )

    # 1. data loaders
    print(
        "Loading shared-bottom datasets from "
        f"{data_cfg['paths']['train_implicit']} and {data_cfg['paths']['train_explicit']}"
    )
    implicit_loader, explicit_loader, validation_loader, test_loader = (
        build_dataloaders(cfg)
    )

    # 2. model, trainer, and training
    model = SharedBottomTwoTower(model_cfg)
    if train_cfg.compile.enabled:
        model = torch.compile(model, mode=train_cfg.compile.mode)
        print(f"Model compiled with torch.compile (mode={train_cfg.compile.mode})")
    trainer = SharedBottomTwoTowerTrainer(model, train_cfg)
    trainer.fit(implicit_loader, explicit_loader, validation_loader)

    # 3. evaluation
    if test_loader is not None:
        test_metrics = trainer.evaluate(test_loader)
        print(f"Final test metrics: {test_metrics}")


if __name__ == "__main__":
    main()

# python -m gerbil_train.cli.shared_bottom_two_tower_train --config configs/experiment/sbtt_ml1m_v1.yaml
