"""Train a MIND model on TFRecord samples."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from gerbil_train.utils.config import load_experiment_config, parse_args
from gerbil_train.utils.run import close_exp_log, create_run_dir, save_run_configs, setup_exp_log
from gerbil_train.utils.training import build_dataloaders, build_model_config
from gerbil_train.config.model_config import DIENModelConfig
from gerbil_train.config.train_config import TrainConfig
from gerbil_train.models.mind import MIND
from gerbil_train.trainer.mind_trainer import MINDTrainer

PROJECT_ROOT = Path(__file__).parent.parent.parent
CONFIG_PATH = PROJECT_ROOT / "configs/12-mind/experiment.yaml"


def main() -> None:
    args = parse_args(CONFIG_PATH)
    exp_cfg: dict[str, Any] = load_experiment_config(args.config)
    data_cfg: dict[str, Any] = exp_cfg["data"]
    model_cfg: DIENModelConfig = build_model_config(exp_cfg, DIENModelConfig)

    run_dir = create_run_dir(PROJECT_ROOT / "checkpoints" / "mind")
    setup_exp_log(run_dir)
    train_cfg: TrainConfig = TrainConfig.from_dict(exp_cfg["train"])
    train_cfg.checkpoint.path = str(run_dir)
    train_cfg.logging.plot_path = str(run_dir)
    print(f"Training config | seed={train_cfg.seed} | epochs={train_cfg.epochs} | batch_size={train_cfg.data.batch_size}")
    print(f"Run dir: {run_dir}")
    print(f"Loading TFRecords from {data_cfg['paths']['tfrecord_root']}")

    train_loader, validation_loader, test_loader = build_dataloaders(data_cfg, model_cfg, train_cfg)
    model = MIND(model_cfg)
    if train_cfg.compile.enabled:
        model = torch.compile(model, mode=train_cfg.compile.mode)
        print(f"Model compiled with torch.compile (mode={train_cfg.compile.mode})")
    trainer = MINDTrainer(model, train_cfg, data_cfg)
    trainer.fit(train_loader, validation_loader, test_loader)

    if test_loader is not None:
        test_metrics = trainer.evaluate(test_loader)
        print(f"Final test metrics: {test_metrics}")
    save_run_configs(args.config, run_dir, project_root=PROJECT_ROOT)
    close_exp_log()


if __name__ == "__main__":
    main()
