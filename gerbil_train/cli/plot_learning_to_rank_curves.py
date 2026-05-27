from __future__ import annotations

import argparse
from pathlib import Path

from gerbil_train.utils.plot import plot_checkpoint_curve_comparisons


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for aggregated curve plotting."""
    parser = argparse.ArgumentParser(
        description="Plot aggregated loss and NDCG curves from checkpoint txt files"
    )
    parser.add_argument(
        "--checkpoints-dir",
        type=Path,
        default=Path("checkpoints"),
        help="Root directory containing saved curve txt files",
    )
    parser.add_argument(
        "--loss-output",
        type=Path,
        default=None,
        help="Output path for the aggregated loss figure",
    )
    parser.add_argument(
        "--ndcg-output",
        type=Path,
        default=None,
        help="Output path for the aggregated NDCG figure",
    )
    return parser.parse_args()


def main() -> None:
    """Generate aggregate loss and NDCG figures for all saved algorithms."""
    args = parse_args()
    loss_output_path = (
        args.loss_output
        if args.loss_output is not None
        else args.checkpoints_dir / "loss_comparison.png"
    )
    ndcg_output_path = (
        args.ndcg_output
        if args.ndcg_output is not None
        else args.checkpoints_dir / "ndcg_comparison.png"
    )

    saved_loss_path, saved_ndcg_path = plot_checkpoint_curve_comparisons(
        checkpoints_dir=args.checkpoints_dir,
        loss_output_path=loss_output_path,
        ndcg_output_path=ndcg_output_path,
    )
    print(f"Saved loss comparison to {saved_loss_path.resolve()}")
    print(f"Saved NDCG comparison to {saved_ndcg_path.resolve()}")


if __name__ == "__main__":
    main()
