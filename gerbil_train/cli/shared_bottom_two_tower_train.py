from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml
from torch.utils.data import DataLoader

from gerbil_train.models.shared_bottom_two_tower import SharedBottomTwoTower
from gerbil_train.data.shared_bottom_two_tower_dataset import SharedBottomTwoTowerExplicitDataset, SharedBottomTwoTowerImplicitDataset
from gerbil_train.trainer.shared_bottom_two_tower_trainer import SBTTTrainer


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_experiment_config(experiment_path: str | Path) -> dict[str, Any]:
    
    exp = load_yaml(experiment_path)

    config = {
        "experiment": exp,
        "data": load_yaml(exp["data"]),
        "model": load_yaml(exp["model"]),
        "train": load_yaml(exp["train"]),
        "eval": load_yaml(exp["eval"]),
        "export": load_yaml(exp["export"]),
    }
    return config


def build_dataloaders(cfg: dict[str, Any]) -> tuple[DataLoader | None, DataLoader | None]:
    data_cfg = cfg["data"]
    loader_cfg = data_cfg["loader"]

    implicit_loader = None
    explicit_loader = None

    if data_cfg.get("sampling", {}).get("implicit", {}).get("negative_sampling", False):
        implicit_dataset = SharedBottomTwoTowerImplicitDataset(
            data_path=data_cfg["paths"]["train_implicit"],
            query_input_dim=cfg["model"]["model"]["query_input_dim"],
            item_input_dim=cfg["model"]["model"]["item_input_dim"],
            num_negatives=data_cfg["sampling"]["implicit"]["num_negatives"],
        )
        implicit_loader = DataLoader(
            implicit_dataset,
            batch_size=loader_cfg["train_batch_size"],
            shuffle=True,
            num_workers=loader_cfg["num_workers"],
            pin_memory=loader_cfg["pin_memory"],
            drop_last=loader_cfg["drop_last"],
        )

    explicit_dataset = SharedBottomTwoTowerExplicitDataset(
        data_path=data_cfg["paths"]["train_explicit"],
        query_input_dim=cfg["model"]["model"]["query_input_dim"],
        item_input_dim=cfg["model"]["model"]["item_input_dim"],
    )
    explicit_loader = DataLoader(
        explicit_dataset,
        batch_size=loader_cfg["train_batch_size"],
        shuffle=True,
        num_workers=loader_cfg["num_workers"],
        pin_memory=loader_cfg["pin_memory"],
        drop_last=loader_cfg["drop_last"],
    )

    return implicit_loader, explicit_loader


def build_model(cfg: dict[str, Any]) -> SharedBottomTwoTower:
    model_cfg = cfg["model"]["model"]

    return SharedBottomTwoTower(
        query_input_dim=model_cfg["query_input_dim"],
        item_input_dim=model_cfg["item_input_dim"],
        shared_hidden_dims=model_cfg["shared_bottom"]["hidden_dims"],
        explicit_hidden_dims=model_cfg["explicit_tower"]["hidden_dims"],
        implicit_hidden_dims=model_cfg["implicit_tower"]["hidden_dims"],
        embedding_dim=model_cfg["embedding_dim"],
        dropout=model_cfg["shared_bottom"].get("dropout", 0.0),
        activation=model_cfg["shared_bottom"].get("activation", "relu"),
        batch_norm=model_cfg["shared_bottom"].get("batch_norm", False),
        normalize_embedding=model_cfg.get("normalize_embedding", False),
    )

def build_trainer(cfg: dict[str, Any], model: SharedBottomTwoTower) -> SBTTTrainer:
    train_cfg = cfg["train"]

    return SBTTTrainer(
        model=model,
        device=train_cfg["device"],
        implicit_lr=train_cfg["optimizer"]["implicit"]["lr"],
        explicit_lr=train_cfg["optimizer"]["explicit"]["lr"],
        weight_decay=train_cfg["optimizer"]["implicit"]["weight_decay"],
        gradient_clip_norm=train_cfg["trainer"].get("gradient_clip_norm"),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to experiment yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_experiment_config(args.config)

    model = build_model(cfg)
    implicit_loader, explicit_loader = build_dataloaders(cfg)
    trainer = build_trainer(cfg, model)

    trainer.fit(
        implicit_loader=implicit_loader,
        explicit_loader=explicit_loader,
        implicit_epochs=cfg["train"]["trainer"]["implicit_epochs"],
        explicit_epochs=cfg["train"]["trainer"]["explicit_epochs"],
    )


if __name__ == "__main__":
    main()