"""Configuration loading helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
import argparse

__all__ = ["load_yaml", "load_experiment_config", "parse_args"]


def parse_args(path: Path) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=path,
    )
    return parser.parse_args()


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file into a dictionary.

    :param path: Path to the YAML file
    :return: Parsed YAML content
    """
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_experiment_config(experiment_path: str | Path) -> dict[str, Any]:
    """Load the top-level experiment config and all referenced sub-configs.

    :param experiment_path: Path to the experiment YAML file
    :return: Dictionary containing the resolved config sections
    """
    exp = load_yaml(experiment_path)
    data = load_yaml(exp["data"])
    model = load_yaml(exp["model"])
    train = load_yaml(exp["train"])
    
    config: dict[str, Any] = {
        "experiment": exp,
        "data": data,
        "model": model,
        "train": train,
    }
    for key in ("eval", "export"):
        if key in exp:
            config[key] = load_yaml(exp[key])
    return config
