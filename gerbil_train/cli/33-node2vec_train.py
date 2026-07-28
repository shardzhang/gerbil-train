"""Train a Node2Vec model on TFRecord samples."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from gerbil_train.utils.config import load_experiment_config, parse_args
from gerbil_train.utils.run import close_exp_log, create_run_dir, save_run_configs, setup_exp_log
from gerbil_train.utils.training import build_dataloaders, build_model_config
from gerbil_train.config.model_config import BaseModelConfig
from gerbil_train.config.train_config import TrainConfig
from gerbil_train.models.node2vec import Node2Vec
from gerbil_train.trainer.node2vec_trainer import Node2VecTrainer

PROJECT_ROOT = Path(__file__).parent.parent.parent
CONFIG_PATH = PROJECT_ROOT / "configs/33-node2vec/experiment.yaml"


def main() -> None:
    args = parse_args(CONFIG_PATH)
    exp_cfg: dict[str, Any] = load_experiment_config(args.config)
    data_cfg: dict[str, Any] = exp_cfg["data"]
    model_cfg: BaseModelConfig = build_model_config(exp_cfg, BaseModelConfig)

    run_dir = create_run_dir(PROJECT_ROOT / "checkpoints" / "node2vec")
    setup_exp_log(run_dir)
    train_cfg: TrainConfig = TrainConfig.from_dict(exp_cfg["train"])
    train_cfg.checkpoint.path = str(run_dir)
    print(f"Training config | seed={train_cfg.seed} | epochs={train_cfg.epochs} | batch_size={train_cfg.data.batch_size}")
    print(f"Run dir: {run_dir}")

    train_loader, validation_loader, test_loader = build_dataloaders(data_cfg, model_cfg, train_cfg)
    model = Node2Vec(model_cfg)
    trainer = Node2VecTrainer(model, train_cfg, data_cfg)
    trainer.fit(train_loader)
    save_run_configs(args.config, run_dir, project_root=PROJECT_ROOT)
    close_exp_log()


if __name__ == "__main__":
    main()
