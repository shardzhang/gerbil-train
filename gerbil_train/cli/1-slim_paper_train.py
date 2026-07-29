"""Train SLIM (paper implementation) via coordinate descent on user-item matrix."""

from __future__ import annotations

from pathlib import Path

from gerbil_train.utils.config import load_experiment_config, parse_args
from gerbil_train.utils.run import close_run_log, create_run_dir, save_run_configs, setup_run_log
from gerbil_train.trainer.slim_paper_trainer import SLIMPaperTrainer

PROJECT_ROOT = Path(__file__).parent.parent.parent
CONFIG_PATH = PROJECT_ROOT / "configs/1-slim_paper/experiment.yaml"


def main() -> None:
    args = parse_args(CONFIG_PATH)
    exp_cfg = load_experiment_config(args.config)
    data_cfg = exp_cfg["data"]
    train_cfg = exp_cfg["train"]

    root = Path(data_cfg["paths"]["tfrecord_root"])
    train_dir = root / data_cfg["split_subdirs"]["train"] / "tfrecord"
    val_dir = root / data_cfg["split_subdirs"]["val"] / "tfrecord"
    test_dir = root / data_cfg["split_subdirs"]["test"] / "tfrecord"

    run_dir = create_run_dir(PROJECT_ROOT / "checkpoints" / "slim_paper")
    setup_run_log(run_dir)
    print(f"Run dir: {run_dir}")
    print(f"Data root: {root}")

    trainer = SLIMPaperTrainer(data_cfg, train_cfg)
    W = trainer.run(
        train_dir=str(train_dir),
        val_dir=str(val_dir),
        test_dir=str(test_dir),
    )

    # Save W matrix
    import scipy.sparse
    w_path = Path(run_dir) / "W.npz"
    scipy.sparse.save_npz(str(w_path), W)
    print(f"W saved to {w_path}")

    """
    W = scipy.sparse.load_npz("path/to/W.npz")
    print(type(W))       # <class 'scipy.sparse._csc.csc_matrix'>
    print(W.shape)       # (3705, 3705)
    print(W.nnz)         # 34378 (非零元素个数)
    """
    save_run_configs(args.config, run_dir, project_root=PROJECT_ROOT)
    close_run_log()


if __name__ == "__main__":
    main()

# python3 -m gerbil_train.cli.1-slim_paper_train --config 1-slim_paper/experiment.yaml
