"""Plotting helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

__all__ = [
    "load_curve_values",
    "plot_checkpoint_curve_comparisons",
    "plot_curve_comparison",
    "save_curve_values",
]


def save_curve_values(values: Sequence[float], path: str | Path) -> None:
    """Save curve values to a plain-text file.

    Each line stores ``epoch_index<TAB>value`` so the file can later be read
    back and used for plotting.

    :param values: Sequence of numeric values ordered by epoch
    :param path: Destination text file path
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for epoch_index, value in enumerate(values, start=1):
            f.write(f"{epoch_index}\t{value:.10f}\n")


def load_curve_values(path: str | Path) -> list[float]:
    """Load curve values from a plain-text file.

    The loader accepts either ``epoch_index<TAB>value`` rows or one numeric
    value per line.

    :param path: Source text file path
    :return: List of curve values ordered by epoch
    """
    path = Path(path)
    values: list[float] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            if "\t" in stripped:
                _, value = stripped.split("\t", 1)
                values.append(float(value))
            else:
                values.append(float(stripped))
    return values
