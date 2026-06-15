from __future__ import annotations

import argparse
from pathlib import Path

from gerbil_train.data.learning_to_rank_dataset import convert_mslr_fold_to_pt


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for preparing MSLR-WEB10K ranking data."""
    parser = argparse.ArgumentParser(
        description="Convert one raw MSLR-WEB10K fold into the grouped .pt format"
    )
    parser.add_argument(
        "--fold-dir",
        type=Path,
        required=True,
        help="Path to one raw fold directory such as MSLR-WEB10K/Fold1",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("data") / "MSLR-WEB10K.pt",
        help="Output path for the grouped .pt dataset",
    )
    return parser.parse_args()


def main() -> None:
    """Convert one raw MSLR-WEB10K fold into the training CLI input format."""
    args = parse_args()
    output_path = convert_mslr_fold_to_pt(args.fold_dir, args.output_path)
    print(f"Saved grouped dataset to {output_path.resolve()}")


if __name__ == "__main__":
    main()
