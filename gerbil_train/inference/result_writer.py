"""Result writing utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

__all__ = ["write_results"]


def write_results(
    results: list[dict[str, Any]],
    path: str | Path,
    fmt: str = "tsv",
) -> None:
    """Write inference results to a file.

    :param results: List of dicts with keys ``user_id``, ``score``, ``label``.
    :param path: Output file path.
    :param fmt: ``"tsv"`` (tab-separated) or ``"json"``.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if fmt == "tsv":
        with path.open("w", encoding="utf-8") as f:
            f.write("user_id\tscore\tlabel\n")
            for r in results:
                f.write(f"{r['user_id']}\t{r['score']:.6f}\t{r['label']}\n")
    elif fmt == "json":
        import json
        with path.open("w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
    else:
        raise ValueError(f"Unsupported format: {fmt}")

    print(f"Wrote {len(results)} results to {path}")
