"""Run directory management for reproducible experiments."""

from __future__ import annotations

import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

_log_file: Any = None
_orig_stdout: Any = None


class _ExpTee:
    """Duplicates writes to multiple file-like objects."""
    def __init__(self, *files):
        self.files = files
    def write(self, data):
        for f in self.files:
            f.write(data)
        self.flush()
    def flush(self):
        for f in self.files:
            f.flush()


def setup_exp_log(run_dir: str | Path) -> None:
    """Redirect stdout to both terminal and exp.log in the run directory."""
    global _log_file, _orig_stdout
    log_path = Path(run_dir) / "exp.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    _log_file = open(log_path, "a", encoding="utf-8")
    _orig_stdout = sys.stdout
    sys.stdout = _ExpTee(sys.stdout, _log_file)


def close_exp_log() -> None:
    """Restore stdout and close the exp.log file."""
    global _log_file, _orig_stdout
    if _orig_stdout is not None:
        sys.stdout = _orig_stdout
        _orig_stdout = None
    if _log_file is not None:
        _log_file.close()
        _log_file = None


def create_run_dir(base_dir: str | Path) -> Path:
    """Create a timestamped run directory.
    Returns the run_dir.
    """
    run_id = datetime.now().strftime("%Y%m%d%H%M%S")
    run_dir = Path(base_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


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
