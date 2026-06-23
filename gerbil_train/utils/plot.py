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


def _extract_curve_label(path: Path, curve_suffix: str) -> str:
    """Extract the algorithm label from a curve filename.

    :param path: Curve text file path
    :param curve_suffix: Expected suffix such as ``loss`` or ``metric``
    :return: Legend label derived from the filename
    """
    stem = path.stem
    suffix = f"_{curve_suffix}"
    if stem.startswith("training_curves_") and stem.endswith(suffix):
        return stem[len("training_curves_") : -len(suffix)]
    if stem.endswith(suffix):
        return stem[: -len(suffix)]
    return stem


def plot_curve_comparison(
    curve_paths: Sequence[str | Path],
    output_path: str | Path,
    *,
    curve_suffix: str,
    title: str,
    ylabel: str,
    normalize_to_first_value: bool = False,
) -> Path:
    """Plot multiple curve text files on the same figure.

    :param curve_paths: Sequence of curve text file paths
    :param output_path: Destination path for the rendered figure
    :param curve_suffix: Suffix used to derive the legend label from each file
    :param title: Figure title
    :param ylabel: Y-axis label
    :param normalize_to_first_value: Whether to divide each curve by its first value
    :return: Saved figure path
    """
    if not curve_paths:
        raise FileNotFoundError(f"No '*_{curve_suffix}.txt' files were found.")

    normalized_paths = [Path(path) for path in curve_paths]
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    sorted_paths = sorted(
        normalized_paths,
        key=lambda path: _extract_curve_label(path, curve_suffix),
    )

    plt.figure(figsize=(10, 6))
    for path in sorted_paths:
        values = load_curve_values(path)
        if normalize_to_first_value and values:
            first_value = values[0]
            if first_value != 0:
                values = [value / first_value for value in values]
        epochs = range(1, len(values) + 1)
        label = _extract_curve_label(path, curve_suffix)
        plt.plot(epochs, values, linewidth=2, label=label)

    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend(title="Algorithm")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    return output_path


def plot_checkpoint_curve_comparisons(
    checkpoints_dir: str | Path,
    *,
    loss_output_path: str | Path | None = None,
    ndcg_output_path: str | Path | None = None,
) -> tuple[Path, Path]:
    """Plot aggregate loss and NDCG figures from checkpoint curve text files.

    The function scans ``checkpoints_dir`` recursively for ``*_loss.txt`` and
    ``*_metric.txt`` files, then saves two figures that compare all algorithms
    on the same axes.

    :param checkpoints_dir: Root directory containing curve text files
    :param loss_output_path: Optional output path for the loss comparison figure
    :param ndcg_output_path: Optional output path for the NDCG comparison figure
    :return: Tuple of ``(loss_figure_path, ndcg_figure_path)``
    """
    checkpoints_dir = Path(checkpoints_dir)
    loss_curve_paths = sorted(checkpoints_dir.rglob("*_loss.txt"))
    metric_curve_paths = sorted(checkpoints_dir.rglob("*_metric.txt"))

    loss_output_path = (
        checkpoints_dir / "loss_comparison.png"
        if loss_output_path is None
        else Path(loss_output_path)
    )
    ndcg_output_path = (
        checkpoints_dir / "ndcg_comparison.png"
        if ndcg_output_path is None
        else Path(ndcg_output_path)
    )

    saved_loss_path = plot_curve_comparison(
        loss_curve_paths,
        loss_output_path,
        curve_suffix="loss",
        title="Normalized Loss Comparison Across Algorithms",
        ylabel="Loss / Epoch-1 Loss",
        normalize_to_first_value=True,
    )
    saved_ndcg_path = plot_curve_comparison(
        metric_curve_paths,
        ndcg_output_path,
        curve_suffix="metric",
        title="Validation NDCG Comparison Across Algorithms",
        ylabel="NDCG",
    )
    return saved_loss_path, saved_ndcg_path
