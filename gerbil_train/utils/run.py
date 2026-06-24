"""Run directory management for reproducible experiments."""

from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path
import yaml


def create_run_dir(base_dir: str | Path) -> tuple[Path, Path, Path]:
    """Create a timestamped run directory.
    Returns ``(run_dir, checkpoint_path, plot_path)`` where
    ``checkpoint_path`` and ``plot_path`` are derived paths inside the run dir.
    """
    run_id = datetime.now().strftime("%Y%m%d%H%M%S")
    run_dir = Path(base_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir, run_dir / "best_model.pth", run_dir / "training_curves.png"


def save_run_configs(experiment_path: str | Path, run_dir: Path, project_root: str | Path | None = None) -> None:
    """Copy the experiment's configuration files into the run directory.

    :param experiment_path: Path to the experiment YAML file.
    :param run_dir: Destination run directory.
    :param project_root: Project root used to resolve relative config paths.
        If ``None``, defaults to two levels above the experiment file.
    """
    exp_cfg_path = Path(experiment_path)
    root = Path(project_root) if project_root is not None else exp_cfg_path.parent.parent
    with open(exp_cfg_path, encoding="utf-8") as f:
        exp_raw = yaml.safe_load(f)

    shutil.copy2(str(exp_cfg_path), str(run_dir / "experiment.yaml"))
    for key in ("data", "model", "train"):
        sub_path = exp_raw.get(key)
        if sub_path:
            src = root / sub_path
            if src.exists():
                shutil.copy2(str(src), str(run_dir / f"{key}.yaml"))
    print(f"Run artifacts saved to {run_dir}")
