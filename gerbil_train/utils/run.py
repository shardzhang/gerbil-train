"""Run directory management for reproducible experiments."""

from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path
from typing import Any


def create_run_dir(base_dir: str | Path) -> tuple[Path, Path, Path]:
    """Create a timestamped run directory.

    Returns ``(run_dir, checkpoint_path, plot_path)`` where
    ``checkpoint_path`` and ``plot_path`` are derived paths inside the run dir.
    """
    run_id = datetime.now().strftime("%Y%m%d%H%M%S")
    run_dir = Path(base_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir, run_dir / "best_model.pth", run_dir / "training_curves.png"


def save_run_configs(
    experiment_path: str | Path,
    run_dir: Path,
    project_root: str | Path | None = None,
) -> None:
    """Copy the experiment's configuration files into the run directory.

    :param experiment_path: Path to the experiment YAML file.
    :param run_dir: Destination run directory.
    :param project_root: Project root used to resolve relative config paths.
        If ``None``, defaults to two levels above the experiment file.
    """
    import yaml

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


def build_field_entries(
    cfg_path: str | Path,
    field_specs: list,
    extra_keys: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], int]:
    """Build field entries from specs and update the model YAML config.

    :param cfg_path: Path to the model YAML config file.
    :param field_specs: List of field spec objects with ``name``, ``index``, ``field_type``, ``dim``.
    :param extra_keys: Optional dict of extra keys to write to the config root (e.g. ``target_size``).
    :return: ``(enabled_entries, default_emb_dim)``
    """
    import yaml
    from gerbil_train.config import GwENFieldEntry

    raw_cfg = yaml.safe_load(Path(cfg_path).read_text(encoding="utf-8"))
    default_emb = int(raw_cfg.get("embedding", {}).get("default_emb_dim", 16))
    existing = raw_cfg.get("embedding", {}).get("fields", {}) or {}

    entries: dict[str, GwENFieldEntry] = {}
    for spec in field_specs:
        ex = existing.get(spec.name, {})
        entries[spec.name] = GwENFieldEntry(
            f_index=spec.index, f_type=spec.field_type, vocab_size=int(spec.dim),
            emb_dim=int(ex.get("emb_dim", default_emb)),
            enabled=bool(ex.get("enabled", True)),
        )

    if extra_keys:
        raw_cfg.update(extra_keys)

    raw_cfg["embedding"]["fields"] = {
        n: {"f_index": e.f_index, "f_type": e.f_type, "vocab_size": e.vocab_size, "emb_dim": e.emb_dim, "enabled": e.enabled}
        for n, e in sorted(entries.items(), key=lambda x: x[1].f_index)
    }
    with open(cfg_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(raw_cfg, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    print(f"Config written to {cfg_path}")

    enabled_entries = {n: e for n, e in entries.items() if e.enabled}
    return enabled_entries, default_emb


def filter_enabled_fields(
    field_specs: list,
    field_enabled: dict[str, bool],
) -> list:
    """Filter field specs based on the ``enabled`` flag per field.

    :param field_specs: List of field spec objects with a ``.name`` attribute.
    :param field_enabled: Mapping ``field_name -> enabled (bool)``.
    :return: List of specs for which ``field_enabled`` allows.
    """
    disabled = [name for name, enabled in field_enabled.items() if not enabled]
    if disabled:
        print(f"Disabled fields ({len(disabled)}): {disabled}")
    return [spec for spec in field_specs if field_enabled.get(spec.name, True)]
